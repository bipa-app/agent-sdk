//! Shared Gemini API types, conversion functions, and SSE stream parser.
//!
//! Used by both the `GeminiProvider` (API key auth) and `VertexProvider` (`OAuth2` Bearer auth)
//! since they share the same request/response format.

use crate::streaming::{StreamBox, StreamDelta, StreamErrorKind};
use agent_sdk_foundation::llm::{Content, ContentBlock, StopReason, Usage};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

// ============================================================================
// API Request Types
// ============================================================================

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiGenerateContentRequest<'a> {
    pub contents: &'a [ApiContent],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<&'a ApiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<&'a [ApiToolConfig]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ApiFunctionCallingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<ApiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<&'a str>,
}

#[derive(Serialize, Deserialize)]
pub struct ApiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Parts can be missing in some edge cases (e.g., empty responses, safety blocks)
    #[serde(default)]
    pub parts: Vec<ApiPart>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ApiPart {
    Text {
        text: String,
        /// Thought signature may appear with text in Gemini 3 models
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: ApiBlob,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: ApiFunctionCall,
        /// Thought signature for Gemini 3 models — preserves reasoning context
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: ApiFunctionResponse,
    },
    /// Catch-all for unknown part types to prevent parse failures
    Unknown(serde_json::Value),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ApiBlob {
    pub mime_type: String,
    pub data: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiFunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiToolConfig {
    pub function_declarations: Vec<ApiFunctionDeclaration>,
}

#[derive(Serialize)]
pub struct ApiFunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Gemini `toolConfig.functionCallingConfig` wire format.
///
/// - `mode: "AUTO"` — model decides.
/// - `mode: "ANY"` with `allowed_function_names` — force specific function(s).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiFunctionCallingConfig {
    pub function_calling_config: ApiFunctionCallingMode,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiFunctionCallingMode {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

impl ApiFunctionCallingConfig {
    pub fn from_tool_choice(tc: &agent_sdk_foundation::llm::ToolChoice) -> Self {
        match tc {
            agent_sdk_foundation::llm::ToolChoice::Auto => Self {
                function_calling_config: ApiFunctionCallingMode {
                    mode: "AUTO".to_owned(),
                    allowed_function_names: None,
                },
            },
            agent_sdk_foundation::llm::ToolChoice::Tool(name) => Self {
                function_calling_config: ApiFunctionCallingMode {
                    mode: "ANY".to_owned(),
                    allowed_function_names: Some(vec![name.clone()]),
                },
            },
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ApiThinkingConfig>,
    /// Native structured-output MIME type. `"application/json"` switches
    /// Gemini into JSON mode; paired with [`response_schema`](Self::response_schema)
    /// the model is constrained to the schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<&'static str>,
    /// Native structured-output schema (`responseSchema`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
}

/// Gemini thinking configuration.
///
/// Gemini 3.x models use `thinking_level` (LOW / MEDIUM / HIGH).
/// Thinking **cannot be disabled** on Gemini 3 Pro and 3.1 Pro.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiThinkingConfig {
    pub thinking_level: &'static str,
}

/// Map an agent-sdk `ThinkingConfig` to the Gemini API thinking level.
///
/// Gemini 3 models use `thinkingLevel` with values:
/// - `MINIMAL`: near-zero thinking (Flash/Flash-Lite only, not Pro)
/// - `LOW`: minimal latency and cost
/// - `MEDIUM`: balanced
/// - `HIGH`: maximum reasoning depth (default, dynamic)
pub const fn map_thinking_config(
    config: &agent_sdk_foundation::llm::ThinkingConfig,
) -> ApiThinkingConfig {
    use agent_sdk_foundation::llm::{Effort, ThinkingMode};
    // If an explicit effort is set, use it directly
    if let Some(effort) = config.effort {
        let level = match effort {
            Effort::Low => "LOW",
            Effort::Medium => "MEDIUM",
            Effort::High | Effort::Max => "HIGH",
        };
        return ApiThinkingConfig {
            thinking_level: level,
        };
    }
    let level = match &config.mode {
        // Adaptive → let the model decide (HIGH = dynamic)
        ThinkingMode::Adaptive => "HIGH",
        // Explicit budget: map to LOW / MEDIUM / HIGH
        ThinkingMode::Enabled { budget_tokens } => {
            if *budget_tokens <= 4_096 {
                "LOW"
            } else if *budget_tokens <= 16_384 {
                "MEDIUM"
            } else {
                "HIGH"
            }
        }
    };
    ApiThinkingConfig {
        thinking_level: level,
    }
}

/// JSON-Schema keywords that Gemini's OpenAPI-flavoured `responseSchema`
/// (and `functionDeclarations.parameters`) rejects.
const GEMINI_STRIPPED_SCHEMA_KEYS: [&str; 6] = [
    "$schema",
    "$id",
    "additionalProperties",
    "title",
    "definitions",
    "$defs",
];

/// Maximum `$ref` resolution depth. Guards against runaway inlining of
/// (self-)recursive schemas, which cannot be represented as a finite inlined
/// document anyway.
const GEMINI_MAX_REF_DEPTH: usize = 64;

/// Schema keywords whose value is a map of arbitrary *member name* → subschema
/// (a `{ name: schema }` object), as opposed to a nested schema-keyword object.
///
/// The member **names** here are user data, not schema keywords: a property can
/// legitimately be called `title`, `additionalProperties`, `enum`, `$ref`, etc.
/// Those names must be preserved verbatim — only each member's subschema *value*
/// is sanitized. Recursing into these maps with the generic keyword-strip pass
/// would silently delete a property literally named `title` (a
/// [`GEMINI_STRIPPED_SCHEMA_KEYS`] entry) while a sibling `required` array still
/// referenced it, which makes Gemini 400 the entire `generateContent` request
/// with `properties[...].required[i]: property is not defined`.
const GEMINI_NAMED_SUBSCHEMA_MAPS: [&str; 2] = ["properties", "patternProperties"];

/// Resolve a local JSON-pointer `$ref` (`#/$defs/Name` or
/// `#/definitions/Name`) to its definition name.
fn resolve_local_ref_name(reference: &str) -> Option<&str> {
    reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))
}

/// Collect the top-level `$defs` / `definitions` map of a schema document.
fn collect_schema_defs(schema: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    let mut defs = serde_json::Map::new();
    if let serde_json::Value::Object(map) = schema {
        for key in ["$defs", "definitions"] {
            if let Some(serde_json::Value::Object(group)) = map.get(key) {
                for (name, def) in group {
                    defs.insert(name.clone(), def.clone());
                }
            }
        }
    }
    defs
}

/// Sanitize one `key: value` entry of a schema object into `out`.
///
/// Shared by both the plain object walk and the `$ref`-merge path so member-name
/// preservation and keyword stripping stay identical on both. Stripped keys
/// (and unresolved `$ref`) are dropped. For a [`GEMINI_NAMED_SUBSCHEMA_MAPS`]
/// key (`properties` / `patternProperties`) the value is a `{ name: subschema }`
/// map whose **names** are preserved verbatim while each subschema is sanitized;
/// every other value is sanitized generically.
fn insert_sanitized_entry(
    out: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    val: &serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
) {
    // `$ref` is dropped here only when it could not be resolved by the caller;
    // leaving it would point at a stripped `$defs` target.
    if key == "$ref" || GEMINI_STRIPPED_SCHEMA_KEYS.contains(&key) {
        return;
    }
    let sanitized = if GEMINI_NAMED_SUBSCHEMA_MAPS.contains(&key)
        && let serde_json::Value::Object(members) = val
    {
        let mut sanitized_members = serde_json::Map::with_capacity(members.len());
        for (member_name, subschema) in members {
            // Preserve the member NAME verbatim; only sanitize its subschema.
            sanitized_members.insert(
                member_name.clone(),
                sanitize_schema_value(subschema, defs, depth),
            );
        }
        serde_json::Value::Object(sanitized_members)
    } else {
        sanitize_schema_value(val, defs, depth)
    };
    out.insert(key.to_owned(), sanitized);
}

fn sanitize_schema_value(
    value: &serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            // A `$ref` node: inline the referenced definition so the wire schema
            // is self-contained (Gemini does not resolve `$ref`/`$defs`).
            if let Some(serde_json::Value::String(reference)) = map.get("$ref")
                && depth < GEMINI_MAX_REF_DEPTH
                && let Some(name) = resolve_local_ref_name(reference)
                && let Some(target) = defs.get(name)
            {
                let mut inlined = sanitize_schema_value(target, defs, depth + 1);
                // Merge any sibling keys that accompanied the `$ref` (e.g. an
                // overriding `description`) on top of the inlined definition.
                if let serde_json::Value::Object(inlined_map) = &mut inlined {
                    for (key, val) in map {
                        insert_sanitized_entry(inlined_map, key.as_str(), val, defs, depth + 1);
                    }
                }
                return inlined;
            }

            let mut out = serde_json::Map::with_capacity(map.len());
            for (key, val) in map {
                insert_sanitized_entry(&mut out, key.as_str(), val, defs, depth);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(|item| sanitize_schema_value(item, defs, depth))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Defensive final pass: at every object level, drop any `required` entry that
/// names a property absent from the sibling `properties` map.
///
/// Belt-and-braces guarantee that no schema — however mangled by upstream
/// conversion — can 400 a `generateContent` call, since Gemini rejects the
/// *whole* request on a single dangling `required` reference. In the normal path
/// the primary [`insert_sanitized_entry`] fix already keeps `properties` and
/// `required` in sync, so this pass is a no-op; it only fires (with a
/// debug-level log) if some other transform ever drops a referenced property.
fn prune_dangling_required(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            // Snapshot the defined property names first (owned, so the immutable
            // borrow ends before we mutate `required`).
            let defined: Option<std::collections::BTreeSet<String>> = map
                .get("properties")
                .and_then(serde_json::Value::as_object)
                .map(|props| props.keys().cloned().collect());
            if let Some(defined) = defined
                && let Some(serde_json::Value::Array(required)) = map.get_mut("required")
            {
                required.retain(|entry| match entry.as_str() {
                    Some(name) if !defined.contains(name) => {
                        log::debug!(
                            "gemini schema sanitize: dropping `required` entry {name:?} with no \
                             matching property after conversion"
                        );
                        false
                    }
                    _ => true,
                });
            }
            for child in map.values_mut() {
                prune_dangling_required(child);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items.iter_mut() {
                prune_dangling_required(item);
            }
        }
        _ => {}
    }
}

/// Sanitize a JSON Schema document into the subset Gemini accepts for
/// `responseSchema` and `functionDeclarations.parameters`.
///
/// Gemini's structured-output schema is OpenAPI-flavoured and rejects several
/// JSON-Schema-isms (`$schema`, `additionalProperties`, `title`, `$id`,
/// `definitions`/`$defs`, `$ref`). schemars-derived schemas emit `$defs` + `$ref`
/// for any nested struct, so this first **inlines** every resolvable local
/// `$ref` against the document's `$defs`/`definitions`, then strips the
/// unsupported keys recursively while preserving the structural keywords
/// (`type`, `properties`, `items`, `required`, `enum`) — including nested
/// `items` and object subschemas, whose member names survive verbatim. The
/// result is a self-contained document with no dangling `$ref` pointing at a
/// removed target. A final [`prune_dangling_required`] pass drops any `required`
/// entry left without a matching property, so no schema can 400 the request.
/// The runtime still validates the *model output* against the original,
/// un-sanitized schema, so sanitization only relaxes what is sent on the wire.
#[must_use]
pub fn gemini_response_schema(schema: &serde_json::Value) -> serde_json::Value {
    let defs = collect_schema_defs(schema);
    let mut sanitized = sanitize_schema_value(schema, &defs, 0);
    prune_dangling_required(&mut sanitized);
    sanitized
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiGenerateContentResponse {
    pub candidates: Vec<ApiCandidate>,
    pub usage_metadata: Option<ApiUsageMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiCandidate {
    /// Content can be absent when the response is blocked by safety filters.
    #[serde(default = "ApiCandidate::empty_content")]
    pub content: ApiContent,
    pub finish_reason: Option<ApiFinishReason>,
}

impl ApiCandidate {
    const fn empty_content() -> ApiContent {
        ApiContent {
            role: None,
            parts: Vec::new(),
        }
    }
}

/// Gemini API finish reasons.
///
/// Unknown variants (e.g. `MALFORMED_FUNCTION_CALL`, `BLOCKLIST`,
/// `PROHIBITED_CONTENT`, `SPII`) are mapped to `Other` via `#[serde(other)]`
/// to prevent deserialization failures.
#[derive(Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApiFinishReason {
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    pub prompt: u32,
    #[serde(default, rename = "candidatesTokenCount")]
    pub candidates: u32,
    #[serde(default, rename = "cachedContentTokenCount")]
    pub cached_content: u32,
}

impl ApiUsageMetadata {
    pub const fn into_usage(self) -> Usage {
        Usage {
            input_tokens: self.prompt,
            output_tokens: self.candidates,
            cached_input_tokens: self.cached_content,
            cache_creation_input_tokens: 0,
        }
    }
}

// ============================================================================
// Conversion Functions
// ============================================================================

pub fn build_api_contents(messages: &[agent_sdk_foundation::llm::Message]) -> Vec<ApiContent> {
    // Build a mapping of tool_use_id -> function_name from all messages
    let mut tool_names: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for msg in messages {
        if let Content::Blocks(blocks) = &msg.content {
            for block in blocks {
                if let ContentBlock::ToolUse { id, name, .. } = block {
                    tool_names.insert(id.clone(), name.clone());
                }
            }
        }
    }

    let mut contents = Vec::new();

    for msg in messages {
        let role = match msg.role {
            agent_sdk_foundation::llm::Role::User => "user",
            agent_sdk_foundation::llm::Role::Assistant => "model",
        };

        let parts = match &msg.content {
            Content::Text(text) => vec![ApiPart::Text {
                text: text.clone(),
                thought_signature: None,
            }],
            Content::Blocks(blocks) => {
                let mut parts = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            parts.push(ApiPart::Text {
                                text: text.clone(),
                                thought_signature: None,
                            });
                        }
                        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
                        ContentBlock::Image { source } | ContentBlock::Document { source } => {
                            // `source.data` is already base64; the Gemini wire
                            // format also expects base64, so pass it through
                            // directly instead of decoding and re-encoding it
                            // (a no-op round-trip) on every turn. Well-formedness
                            // is enforced once by `validate_request_attachments`.
                            parts.push(ApiPart::InlineData {
                                inline_data: ApiBlob {
                                    mime_type: source.media_type.clone(),
                                    data: source.data.clone(),
                                },
                            });
                        }
                        ContentBlock::ToolUse {
                            id: _,
                            name,
                            input,
                            thought_signature,
                        } => {
                            parts.push(ApiPart::FunctionCall {
                                function_call: ApiFunctionCall {
                                    name: name.clone(),
                                    args: input.clone(),
                                },
                                thought_signature: thought_signature.clone(),
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let func_name = tool_names
                                .get(tool_use_id)
                                .cloned()
                                .unwrap_or_else(|| "unknown_function".to_owned());
                            let response = if is_error.unwrap_or(false) {
                                serde_json::json!({ "error": content })
                            } else {
                                serde_json::json!({ "result": content })
                            };
                            parts.push(ApiPart::FunctionResponse {
                                function_response: ApiFunctionResponse {
                                    name: func_name,
                                    response,
                                },
                            });
                        }
                        // `ContentBlock` is `#[non_exhaustive]`; a block kind this
                        // SDK version cannot map to a Gemini part is skipped.
                        _ => {
                            log::warn!("Skipping unrecognized Gemini content block");
                        }
                    }
                }
                parts
            }
        };

        contents.push(ApiContent {
            role: Some(role.to_owned()),
            parts,
        });
    }

    contents
}

pub fn convert_tools_to_config(tools: Vec<agent_sdk_foundation::llm::Tool>) -> ApiToolConfig {
    ApiToolConfig {
        function_declarations: tools
            .into_iter()
            .map(|t| ApiFunctionDeclaration {
                name: t.name,
                description: t.description,
                // Gemini rejects the same JSON-Schema-isms in tool parameter
                // schemas as in `responseSchema`; sanitize/inline so nested
                // (`$defs` + `$ref`) tool schemas do not 400 the request.
                parameters: gemini_response_schema(&t.input_schema),
            })
            .collect(),
    }
}

pub fn build_content_blocks(content: &ApiContent) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    for part in &content.parts {
        match part {
            ApiPart::Text { text, .. } => {
                if !text.is_empty() {
                    blocks.push(ContentBlock::Text { text: text.clone() });
                }
            }
            ApiPart::FunctionCall {
                function_call,
                thought_signature,
            } => {
                let id = format!("call_{}", uuid_simple());
                blocks.push(ContentBlock::ToolUse {
                    id,
                    name: function_call.name.clone(),
                    input: function_call.args.clone(),
                    thought_signature: thought_signature.clone(),
                });
            }
            ApiPart::InlineData { .. } | ApiPart::FunctionResponse { .. } => {
                // Inline media parts and function responses are input-only in our current SDK flow.
            }
            ApiPart::Unknown(value) => {
                log::warn!("Unknown API part type in Gemini response, skipping part={value:?}");
            }
        }
    }

    blocks
}

pub fn uuid_simple() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Map an `ApiFinishReason` to a `StopReason`, overriding to `ToolUse` when tool calls are present.
pub const fn map_finish_reason(reason: &ApiFinishReason, has_tool_calls: bool) -> StopReason {
    if has_tool_calls {
        StopReason::ToolUse
    } else {
        match reason {
            ApiFinishReason::Stop | ApiFinishReason::Other => StopReason::EndTurn,
            ApiFinishReason::MaxTokens => StopReason::MaxTokens,
            ApiFinishReason::Safety | ApiFinishReason::Recitation => StopReason::StopSequence,
        }
    }
}

// ============================================================================
// Shared SSE Stream Parser
// ============================================================================

/// Truncate SSE data to a bounded preview for log messages.
fn preview_gemini_sse_data(data: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 200;
    let mut preview = data.chars().take(MAX_PREVIEW_CHARS).collect::<String>();
    if data.chars().count() > MAX_PREVIEW_CHARS {
        preview.push('…');
    }
    preview
}

/// Extract an error message from a Gemini mid-stream `{"error": ...}` payload.
///
/// Gemini can interleave an error object into the SSE stream (e.g. quota
/// exhaustion, internal failure) instead of a normal candidate. Returns the
/// human-readable message when the payload is such an error, otherwise `None`.
fn gemini_stream_error(data: &str) -> Option<(String, StreamErrorKind)> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    let error = value.get("error")?;
    let message = error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .map_or_else(|| error.to_string(), str::to_owned);

    // An in-band error arrives after the response headers, so a rate limit
    // reported this way states its delay only in the payload's
    // `google.rpc.RetryInfo` detail. Anything other than a quota rejection
    // stays a (retriable) server error.
    let code = error.get("code").and_then(serde_json::Value::as_i64);
    let status = error.get("status").and_then(serde_json::Value::as_str);
    let kind = if code == Some(429) || status == Some("RESOURCE_EXHAUSTED") {
        StreamErrorKind::RateLimited(crate::retry_hints::google_retry_delay(data))
    } else {
        StreamErrorKind::ServerError
    };
    Some((message, kind))
}

/// Classification of a single Gemini SSE line.
enum GeminiLineParse {
    /// Blank line, comment, or a non-`data:` line — ignore.
    Skip,
    /// A well-formed `generateContent` chunk.
    Response(ApiGenerateContentResponse),
    /// A mid-stream `{"error": ...}` payload, classified by its error code.
    Error {
        message: String,
        kind: StreamErrorKind,
    },
    /// A `data:` line whose JSON could not be parsed.
    ParseFailed { error: String, preview: String },
}

/// Parse one Gemini SSE line into a [`GeminiLineParse`].
fn parse_gemini_sse_line(line: &str) -> GeminiLineParse {
    let line = line.trim();
    if line.is_empty() {
        return GeminiLineParse::Skip;
    }
    // Gemini SSE format: "data: {...}"
    let Some(data) = line.strip_prefix("data: ") else {
        return GeminiLineParse::Skip;
    };
    serde_json::from_str::<ApiGenerateContentResponse>(data).map_or_else(
        |error| {
            gemini_stream_error(data).map_or_else(
                || GeminiLineParse::ParseFailed {
                    error: error.to_string(),
                    preview: preview_gemini_sse_data(data),
                },
                |(message, kind)| GeminiLineParse::Error { message, kind },
            )
        },
        GeminiLineParse::Response,
    )
}

/// Decide the terminal `StreamDelta` for a Gemini stream.
///
/// Mirrors the Anthropic path: a stream that ends without ever seeing a
/// `finishReason` was truncated mid-response and is surfaced as a
/// [`StreamErrorKind::ServerError`] rather than a clean `Done`, so the agent
/// loop does not persist a partial completion as success.
fn gemini_stream_terminal(saw_finish_reason: bool, stop_reason: Option<StopReason>) -> StreamDelta {
    if saw_finish_reason {
        StreamDelta::Done { stop_reason }
    } else {
        log::warn!(
            "Gemini SSE stream ended without a finishReason - stream may have been truncated"
        );
        StreamDelta::Error {
            message: "Stream ended unexpectedly without completion".to_string(),
            kind: StreamErrorKind::ServerError,
        }
    }
}

/// Mutable accumulator state threaded through the Gemini SSE stream parser.
#[derive(Default)]
struct GeminiStreamState {
    block_index: usize,
    in_text_block: bool,
    saw_function_call: bool,
    saw_finish_reason: bool,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
}

/// Fold one parsed Gemini `generateContent` chunk into the stream state,
/// returning the [`StreamDelta`]s the generator should emit for it (in order).
///
/// Extracted from the streaming generator so the `yield`-driven loop stays
/// small: all non-`yield` event mapping lives here and the generator simply
/// re-emits the returned deltas.
fn process_gemini_response(
    state: &mut GeminiStreamState,
    response: ApiGenerateContentResponse,
) -> Vec<StreamDelta> {
    let mut deltas = Vec::new();

    if let Some(usage) = response.usage_metadata {
        state.usage = Some(usage.into_usage());
    }

    let Some(candidate) = response.candidates.into_iter().next() else {
        return deltas;
    };

    if let Some(reason) = &candidate.finish_reason {
        state.saw_finish_reason = true;
        state.stop_reason = Some(map_finish_reason(reason, false));
    }

    // content may be empty on safety-blocked responses
    for part in &candidate.content.parts {
        match part {
            ApiPart::Text { text, .. } if !text.is_empty() => {
                // Gemini sends complete text parts per SSE event (not
                // incremental deltas like Anthropic). Keep the same block_index
                // for consecutive text parts so the StreamAccumulator appends
                // them into one text block.
                if !state.in_text_block {
                    state.in_text_block = true;
                }
                deltas.push(StreamDelta::TextDelta {
                    delta: text.clone(),
                    block_index: state.block_index,
                });
            }
            ApiPart::FunctionCall {
                function_call,
                thought_signature,
            } => {
                // Switching away from text — advance index.
                if state.in_text_block {
                    state.in_text_block = false;
                    state.block_index += 1;
                }
                state.saw_function_call = true;
                let id = format!("call_{}", uuid_simple());
                deltas.push(StreamDelta::ToolUseStart {
                    id: id.clone(),
                    name: function_call.name.clone(),
                    block_index: state.block_index,
                    thought_signature: thought_signature.clone(),
                });
                deltas.push(StreamDelta::ToolInputDelta {
                    id,
                    delta: serde_json::to_string(&function_call.args).unwrap_or_default(),
                    block_index: state.block_index,
                });
                state.block_index += 1;
            }
            _ => {}
        }
    }

    deltas
}

/// Parse a Gemini SSE response stream into `StreamDelta` events.
///
/// This is used by both `GeminiProvider` and `VertexProvider` since the
/// streaming response format is identical. Each SSE event contains independent,
/// incremental text — not cumulative.
pub fn stream_gemini_response(response: reqwest::Response) -> StreamBox<'static> {
    Box::pin(async_stream::stream! {
        let mut state = GeminiStreamState::default();
        let mut buffer = String::new();
        let mut stream = response.bytes_stream();

        loop {
            let eof = match stream.next().await {
                Some(Ok(chunk)) => {
                    buffer.push_str(&String::from_utf8_lossy(&chunk));
                    false
                }
                Some(Err(e)) => {
                    // Include the underlying cause so RefreshingProvider's 401
                    // detection (and operators) can see the real failure.
                    yield Err(anyhow::anyhow!("stream error: {e}"));
                    return;
                }
                None => true,
            };

            // At EOF, flush any residual final line that lacks a trailing
            // newline — it typically carries finishReason / usageMetadata.
            if eof && !buffer.is_empty() && !buffer.ends_with('\n') {
                buffer.push('\n');
            }

            while let Some(pos) = buffer.find('\n') {
                // Parse from the borrowed slice into owned data, then drain
                // once (no per-line reallocation of the residual buffer).
                let parsed = parse_gemini_sse_line(&buffer[..pos]);
                buffer.drain(..=pos);

                match parsed {
                    GeminiLineParse::Skip => {}
                    GeminiLineParse::ParseFailed { error, preview } => {
                        log::warn!(
                            "Failed to parse Gemini SSE event error={error} data_preview={preview}"
                        );
                    }
                    GeminiLineParse::Error { message, kind } => {
                        log::warn!("Gemini stream returned an error payload: {message}");
                        yield Ok(StreamDelta::Error {
                            message,
                            kind,
                        });
                        return;
                    }
                    GeminiLineParse::Response(resp) => {
                        for delta in process_gemini_response(&mut state, resp) {
                            yield Ok(delta);
                        }
                    }
                }
            }

            if eof {
                break;
            }
        }

        // Override to ToolUse if we saw any function calls during the stream.
        if state.saw_function_call {
            state.stop_reason = Some(StopReason::ToolUse);
        }

        // Emit usage (token accounting) before the terminal event.
        if let Some(u) = state.usage {
            yield Ok(StreamDelta::Usage(u));
        }
        yield Ok(gemini_stream_terminal(
            state.saw_finish_reason,
            state.stop_reason,
        ));
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===================
    // API Type Serialization Tests
    // ===================

    #[test]
    fn test_api_content_serialization() {
        let content = ApiContent {
            role: Some("user".to_string()),
            parts: vec![ApiPart::Text {
                text: "Hello!".to_string(),
                thought_signature: None,
            }],
        };

        let json = serde_json::to_string(&content).unwrap_or_default();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"text\":\"Hello!\""));
    }

    #[test]
    fn test_api_part_text_serialization() {
        let part = ApiPart::Text {
            text: "Hello, world!".to_string(),
            thought_signature: None,
        };

        let json = serde_json::to_string(&part).unwrap_or_default();
        assert!(json.contains("\"text\":\"Hello, world!\""));
    }

    #[test]
    fn test_api_part_function_call_serialization() {
        let part = ApiPart::FunctionCall {
            function_call: ApiFunctionCall {
                name: "read_file".to_string(),
                args: serde_json::json!({"path": "/test.txt"}),
            },
            thought_signature: None,
        };

        let json = serde_json::to_string(&part).unwrap_or_default();
        assert!(json.contains("\"functionCall\""));
        assert!(json.contains("\"name\":\"read_file\""));
        assert!(json.contains("\"args\""));
    }

    #[test]
    fn test_api_part_function_response_serialization() {
        let part = ApiPart::FunctionResponse {
            function_response: ApiFunctionResponse {
                name: "read_file".to_string(),
                response: serde_json::json!({"result": "file contents"}),
            },
        };

        let json = serde_json::to_string(&part).unwrap_or_default();
        assert!(json.contains("\"functionResponse\""));
        assert!(json.contains("\"name\":\"read_file\""));
        assert!(json.contains("\"response\""));
    }

    #[test]
    fn test_api_tool_config_serialization() {
        let config = ApiToolConfig {
            function_declarations: vec![ApiFunctionDeclaration {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }],
        };

        let json = serde_json::to_string(&config).unwrap_or_default();
        assert!(json.contains("\"functionDeclarations\""));
        assert!(json.contains("\"name\":\"test_tool\""));
        assert!(json.contains("\"description\":\"A test tool\""));
    }

    #[test]
    fn test_api_generation_config_serialization() {
        let config = ApiGenerationConfig {
            max_output_tokens: Some(1024),
            thinking_config: None,
            response_mime_type: None,
            response_schema: None,
        };

        let json = serde_json::to_string(&config).unwrap_or_default();
        assert!(json.contains("\"maxOutputTokens\":1024"));
        assert!(!json.contains("thinkingConfig"));
        assert!(!json.contains("responseSchema"));
    }

    #[test]
    fn test_api_generation_config_with_thinking() {
        let config = ApiGenerationConfig {
            max_output_tokens: Some(65536),
            thinking_config: Some(ApiThinkingConfig {
                thinking_level: "HIGH",
            }),
            response_mime_type: None,
            response_schema: None,
        };

        let json = serde_json::to_string(&config).unwrap_or_default();
        assert!(json.contains("\"thinkingConfig\""));
        assert!(json.contains("\"thinkingLevel\":\"HIGH\""));
    }

    #[test]
    fn test_generation_config_serializes_response_schema() {
        let config = ApiGenerationConfig {
            max_output_tokens: Some(1024),
            thinking_config: None,
            response_mime_type: Some("application/json"),
            response_schema: Some(serde_json::json!({"type": "object"})),
        };

        let json = serde_json::to_value(&config).unwrap_or_default();
        assert_eq!(json["responseMimeType"], "application/json");
        assert_eq!(json["responseSchema"]["type"], "object");
    }

    #[test]
    fn test_gemini_response_schema_strips_unsupported_keys() {
        let schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Person",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "name": { "type": "string", "title": "Name" }
            },
            "required": ["name"]
        });

        let sanitized = gemini_response_schema(&schema);

        assert!(sanitized.get("$schema").is_none());
        assert!(sanitized.get("title").is_none());
        assert!(sanitized.get("additionalProperties").is_none());
        // Structural keywords are preserved, recursively.
        assert_eq!(sanitized["type"], "object");
        assert_eq!(sanitized["properties"]["name"]["type"], "string");
        assert!(sanitized["properties"]["name"].get("title").is_none());
        assert_eq!(sanitized["required"][0], "name");
    }

    #[test]
    fn test_generate_content_request_serializes_cached_content() {
        let contents = vec![ApiContent {
            role: Some("user".to_string()),
            parts: vec![ApiPart::Text {
                text: "Hello".to_string(),
                thought_signature: None,
            }],
        }];
        let request = ApiGenerateContentRequest {
            contents: &contents,
            system_instruction: None,
            tools: None,
            tool_config: None,
            generation_config: None,
            cached_content: Some("cachedContents/example"),
        };

        let json = serde_json::to_string(&request).unwrap_or_default();
        assert!(json.contains("\"cachedContent\":\"cachedContents/example\""));
    }

    // ===================
    // API Type Deserialization Tests
    // ===================

    #[test]
    fn test_api_response_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello!"}]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50
            }
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        assert_eq!(response.candidates.len(), 1);
        assert!(response.usage_metadata.is_some());
        let usage = response.usage_metadata.unwrap_or(ApiUsageMetadata {
            prompt: 0,
            candidates: 0,
            cached_content: 0,
        });
        assert_eq!(usage.prompt, 100);
        assert_eq!(usage.candidates, 50);
    }

    #[test]
    fn test_usage_metadata_into_usage() {
        let usage = ApiUsageMetadata {
            prompt: 120,
            candidates: 30,
            cached_content: 40,
        }
        .into_usage();

        assert_eq!(usage.input_tokens, 120);
        assert_eq!(usage.output_tokens, 30);
        assert_eq!(usage.cached_input_tokens, 40);
        assert_eq!(usage.cache_creation_input_tokens, 0);
    }

    #[test]
    fn test_api_response_with_function_call_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {"path": "test.txt"}
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP"
                }
            ]
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        let content = &response.candidates[0].content;
        assert_eq!(content.parts.len(), 1);
        match &content.parts[0] {
            ApiPart::FunctionCall { function_call, .. } => {
                assert_eq!(function_call.name, "read_file");
            }
            _ => panic!("Expected FunctionCall part"),
        }
    }

    #[test]
    fn test_api_finish_reason_deserialization() {
        let stop: ApiFinishReason =
            serde_json::from_str("\"STOP\"").unwrap_or_else(|e| panic!("parse failed: {e}"));
        let max_tokens: ApiFinishReason =
            serde_json::from_str("\"MAX_TOKENS\"").unwrap_or_else(|e| panic!("parse failed: {e}"));
        let safety: ApiFinishReason =
            serde_json::from_str("\"SAFETY\"").unwrap_or_else(|e| panic!("parse failed: {e}"));

        assert!(matches!(stop, ApiFinishReason::Stop));
        assert!(matches!(max_tokens, ApiFinishReason::MaxTokens));
        assert!(matches!(safety, ApiFinishReason::Safety));
    }

    #[test]
    fn test_api_finish_reason_unknown_variants_map_to_other() {
        let malformed: ApiFinishReason = serde_json::from_str("\"MALFORMED_FUNCTION_CALL\"")
            .unwrap_or_else(|e| panic!("parse failed: {e}"));
        let blocklist: ApiFinishReason =
            serde_json::from_str("\"BLOCKLIST\"").unwrap_or_else(|e| panic!("parse failed: {e}"));
        let prohibited: ApiFinishReason = serde_json::from_str("\"PROHIBITED_CONTENT\"")
            .unwrap_or_else(|e| panic!("parse failed: {e}"));
        let spii: ApiFinishReason =
            serde_json::from_str("\"SPII\"").unwrap_or_else(|e| panic!("parse failed: {e}"));

        assert!(matches!(malformed, ApiFinishReason::Other));
        assert!(matches!(blocklist, ApiFinishReason::Other));
        assert!(matches!(prohibited, ApiFinishReason::Other));
        assert!(matches!(spii, ApiFinishReason::Other));
    }

    #[test]
    fn test_api_candidate_missing_content_defaults_to_empty() {
        let json = r#"{
            "finishReason": "SAFETY"
        }"#;

        let candidate: ApiCandidate =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        assert!(candidate.content.parts.is_empty());
        assert!(matches!(
            candidate.finish_reason,
            Some(ApiFinishReason::Safety)
        ));
    }

    #[test]
    fn test_api_response_with_unknown_finish_reason_parses() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "I could not call that function."}]
                    },
                    "finishReason": "MALFORMED_FUNCTION_CALL"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 20
            }
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        assert_eq!(response.candidates.len(), 1);
        assert!(matches!(
            response.candidates[0].finish_reason,
            Some(ApiFinishReason::Other)
        ));
    }

    // ===================
    // Message Conversion Tests
    // ===================

    #[test]
    fn test_build_api_contents_simple() {
        let messages = vec![agent_sdk_foundation::llm::Message::user("Hello")];

        let contents = build_api_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("user".to_string()));
        assert_eq!(contents[0].parts.len(), 1);
    }

    #[test]
    fn test_build_api_contents_assistant() {
        let messages = vec![agent_sdk_foundation::llm::Message::assistant("Hi there!")];

        let contents = build_api_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("model".to_string()));
    }

    #[test]
    fn test_convert_tools_to_config() {
        let tools = vec![agent_sdk_foundation::llm::Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
            display_name: "Test Tool".to_string(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        }];

        let api_tools = convert_tools_to_config(tools);
        assert_eq!(api_tools.function_declarations.len(), 1);
        assert_eq!(api_tools.function_declarations[0].name, "test_tool");
    }

    #[test]
    fn test_build_content_blocks_text_only() {
        let content = ApiContent {
            role: Some("model".to_string()),
            parts: vec![ApiPart::Text {
                text: "Hello!".to_string(),
                thought_signature: None,
            }],
        };

        let blocks = build_content_blocks(&content);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello!"));
    }

    #[test]
    fn test_build_content_blocks_with_function_call() {
        let content = ApiContent {
            role: Some("model".to_string()),
            parts: vec![ApiPart::FunctionCall {
                function_call: ApiFunctionCall {
                    name: "read_file".to_string(),
                    args: serde_json::json!({"path": "test.txt"}),
                },
                thought_signature: None,
            }],
        };

        let blocks = build_content_blocks(&content);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::ToolUse { name, .. } if name == "read_file"));
    }

    #[test]
    fn test_uuid_simple_generates_unique_ids() {
        let mut ids = std::collections::HashSet::new();
        for _ in 0..1000 {
            let id = uuid_simple();
            assert!(!id.is_empty());
            assert!(ids.insert(id), "Duplicate ID generated");
        }
        assert_eq!(ids.len(), 1000);
    }

    // ===================
    // Streaming Response Tests
    // ===================

    #[test]
    fn test_streaming_response_text_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}]
                    }
                }
            ]
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        assert_eq!(response.candidates.len(), 1);
        match &response.candidates[0].content.parts[0] {
            ApiPart::Text { text, .. } => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text part"),
        }
    }

    #[test]
    fn test_streaming_response_with_usage_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        let usage = response.usage_metadata.unwrap_or(ApiUsageMetadata {
            prompt: 0,
            candidates: 0,
            cached_content: 0,
        });
        assert_eq!(usage.prompt, 10);
        assert_eq!(usage.candidates, 5);
        assert!(matches!(
            response.candidates[0].finish_reason,
            Some(ApiFinishReason::Stop)
        ));
    }

    #[test]
    fn test_streaming_response_function_call_deserialization() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "NYC"}
                            }
                        }]
                    }
                }
            ]
        }"#;

        let response: ApiGenerateContentResponse =
            serde_json::from_str(json).unwrap_or_else(|e| panic!("parse failed: {e}"));
        match &response.candidates[0].content.parts[0] {
            ApiPart::FunctionCall { function_call, .. } => {
                assert_eq!(function_call.name, "get_weather");
                assert_eq!(function_call.args["location"], "NYC");
            }
            _ => panic!("Expected FunctionCall part"),
        }
    }

    // ===================
    // Finish Reason Mapping Tests
    // ===================

    #[test]
    fn test_map_finish_reason_stop() {
        assert_eq!(
            map_finish_reason(&ApiFinishReason::Stop, false),
            StopReason::EndTurn
        );
    }

    #[test]
    fn test_map_finish_reason_overrides_to_tool_use() {
        assert_eq!(
            map_finish_reason(&ApiFinishReason::Stop, true),
            StopReason::ToolUse
        );
    }

    #[test]
    fn test_map_finish_reason_max_tokens() {
        assert_eq!(
            map_finish_reason(&ApiFinishReason::MaxTokens, false),
            StopReason::MaxTokens
        );
    }

    // ===================
    // SSE Line Parser Tests
    // ===================

    #[test]
    fn test_parse_gemini_sse_line_valid_response() {
        let line = r#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}]}"#;
        match parse_gemini_sse_line(line) {
            GeminiLineParse::Response(resp) => assert_eq!(resp.candidates.len(), 1),
            _ => panic!("expected Response"),
        }
    }

    #[test]
    fn test_parse_gemini_sse_line_skips_blank_and_non_data() {
        assert!(matches!(parse_gemini_sse_line(""), GeminiLineParse::Skip));
        assert!(matches!(
            parse_gemini_sse_line("event: ping"),
            GeminiLineParse::Skip
        ));
    }

    #[test]
    fn test_parse_gemini_sse_line_detects_error_payload() {
        // Mid-stream error payloads must surface as an error, not be silently
        // skipped (which previously let a failed stream look successful).
        let line = r#"data: {"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}"#;
        match parse_gemini_sse_line(line) {
            GeminiLineParse::Error { message, kind } => {
                assert_eq!(message, "quota exceeded");
                // A 429 with no `RetryInfo` detail is a rate limit with no hint.
                assert_eq!(kind, StreamErrorKind::RateLimited(None));
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_parse_gemini_sse_line_reads_in_band_retry_info() {
        // The stream opened with HTTP 200, so the 429's delay is only in the
        // payload: the decoder must classify it as a rate limit and keep the
        // `RetryInfo` delay instead of flattening it to a server error.
        let line = r#"data: {"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"27s"}]}}"#;
        match parse_gemini_sse_line(line) {
            GeminiLineParse::Error { message, kind } => {
                assert_eq!(message, "quota exceeded");
                assert_eq!(
                    kind,
                    StreamErrorKind::RateLimited(Some(std::time::Duration::from_secs(27)))
                );
                assert!(kind.is_recoverable());
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_parse_gemini_sse_line_non_quota_error_stays_a_server_error() {
        let line = r#"data: {"error":{"code":500,"message":"internal","status":"INTERNAL"}}"#;
        match parse_gemini_sse_line(line) {
            GeminiLineParse::Error { message, kind } => {
                assert_eq!(message, "internal");
                assert_eq!(kind, StreamErrorKind::ServerError);
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_parse_gemini_sse_line_reports_malformed_json() {
        let line = "data: {not valid json";
        match parse_gemini_sse_line(line) {
            GeminiLineParse::ParseFailed { error, preview } => {
                assert!(!error.is_empty());
                assert!(preview.contains("not valid json"));
            }
            _ => panic!("expected ParseFailed"),
        }
    }

    #[test]
    fn test_gemini_stream_error_extracts_message_and_kind() {
        assert_eq!(
            gemini_stream_error(r#"{"error":{"message":"boom"}}"#),
            Some(("boom".to_string(), StreamErrorKind::ServerError))
        );
        assert_eq!(gemini_stream_error(r#"{"candidates":[]}"#), None);
    }

    #[test]
    fn test_gemini_stream_terminal_done_when_finish_seen() {
        let terminal = gemini_stream_terminal(true, Some(StopReason::EndTurn));
        assert!(matches!(
            terminal,
            StreamDelta::Done {
                stop_reason: Some(StopReason::EndTurn)
            }
        ));
    }

    #[test]
    fn test_gemini_stream_terminal_error_when_no_finish_reason() {
        // A stream that ends without a finishReason was truncated; surface a
        // ServerError instead of a clean Done so the loop does not persist a
        // partial completion as success.
        let terminal = gemini_stream_terminal(false, None);
        assert!(matches!(
            terminal,
            StreamDelta::Error {
                kind: StreamErrorKind::ServerError,
                ..
            }
        ));
    }

    // ===================
    // Response Schema Sanitizer Tests ($ref inlining)
    // ===================

    #[test]
    fn test_gemini_response_schema_inlines_nested_refs() {
        // Mirrors a schemars-derived schema for a struct with a nested struct:
        // the nested type lands in `$defs` and is referenced via `$ref`.
        let schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Person",
            "type": "object",
            "$defs": {
                "Address": {
                    "title": "Address",
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "city": { "type": "string", "title": "City" }
                    },
                    "required": ["city"]
                }
            },
            "properties": {
                "name": { "type": "string" },
                "home": { "$ref": "#/$defs/Address" }
            },
            "required": ["name", "home"]
        });

        let sanitized = gemini_response_schema(&schema);
        let serialized = serde_json::to_string(&sanitized).unwrap_or_default();

        // No JSON-Schema-isms or dangling refs survive on the wire.
        assert!(!serialized.contains("$ref"), "ref survived: {serialized}");
        assert!(!serialized.contains("$defs"), "defs survived: {serialized}");
        assert!(!serialized.contains("$schema"));
        assert!(!serialized.contains("additionalProperties"));
        assert!(!serialized.contains("title"));

        // The nested definition was inlined in place of the `$ref`.
        assert_eq!(sanitized["properties"]["home"]["type"], "object");
        assert_eq!(
            sanitized["properties"]["home"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(sanitized["properties"]["home"]["required"][0], "city");
    }

    #[test]
    fn test_gemini_response_schema_inlines_array_item_refs() {
        let schema = serde_json::json!({
            "type": "object",
            "definitions": {
                "Tag": { "type": "string", "enum": ["a", "b"] }
            },
            "properties": {
                "tags": { "type": "array", "items": { "$ref": "#/definitions/Tag" } }
            }
        });

        let sanitized = gemini_response_schema(&schema);
        let serialized = serde_json::to_string(&sanitized).unwrap_or_default();
        assert!(!serialized.contains("$ref"));
        assert!(!serialized.contains("definitions"));
        assert_eq!(sanitized["properties"]["tags"]["items"]["type"], "string");
        assert_eq!(sanitized["properties"]["tags"]["items"]["enum"][0], "a");
    }

    #[test]
    fn test_gemini_response_schema_recursive_ref_terminates() {
        // A self-referential schema cannot be fully inlined; the depth guard
        // must terminate and drop the unresolved ref rather than loop forever.
        let schema = serde_json::json!({
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": { "next": { "$ref": "#/$defs/Node" } }
                }
            },
            "properties": { "root": { "$ref": "#/$defs/Node" } }
        });

        let sanitized = gemini_response_schema(&schema);
        let serialized = serde_json::to_string(&sanitized).unwrap_or_default();
        assert!(!serialized.contains("$defs"));
        // Inlining stops at the depth limit; the deepest residual ref is dropped.
        assert_eq!(sanitized["properties"]["root"]["type"], "object");
    }

    #[test]
    fn test_convert_tools_to_config_sanitizes_parameters() {
        let tools = vec![agent_sdk_foundation::llm::Tool {
            name: "make".to_string(),
            description: "build".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "$defs": { "Part": { "type": "string" } },
                "properties": { "part": { "$ref": "#/$defs/Part" } },
                "additionalProperties": false
            }),
            display_name: "Make".to_string(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        }];

        let config = convert_tools_to_config(tools);
        let params = &config.function_declarations[0].parameters;
        let serialized = serde_json::to_string(params).unwrap_or_default();
        assert!(!serialized.contains("$ref"));
        assert!(!serialized.contains("$defs"));
        assert!(!serialized.contains("additionalProperties"));
        assert_eq!(params["properties"]["part"]["type"], "string");
    }

    /// Regression for the live Gemini 400
    /// `properties[issues].items.required[1]: property is not defined`.
    ///
    /// Mirrors `linear_plan_apply`'s `issues` array: an array whose `items`
    /// object carries both `properties` and `required`, where one required
    /// property is literally named `title` — a JSON-Schema annotation keyword
    /// that the keyword-strip pass must NOT delete when it is a property name.
    #[test]
    fn test_convert_tools_preserves_nested_array_item_properties() {
        let tools = vec![agent_sdk_foundation::llm::Tool {
            name: "linear_plan_apply".to_string(),
            description: "apply a plan".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["summary"],
                "properties": {
                    "summary": { "type": "string" },
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["ref", "title", "what_to_build", "verification_criteria"],
                            "properties": {
                                "ref": { "type": "string", "description": "local handle" },
                                "title": { "type": "string" },
                                "what_to_build": { "type": "string" },
                                "verification_criteria": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": { "type": "string" }
                                }
                            }
                        }
                    }
                }
            }),
            display_name: "Plan Apply".to_string(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        }];

        let config = convert_tools_to_config(tools);
        let params = &config.function_declarations[0].parameters;
        let item_props = &params["properties"]["issues"]["items"]["properties"];

        // Every property survives conversion, including the one named `title`.
        for name in ["ref", "title", "what_to_build", "verification_criteria"] {
            assert!(
                item_props.get(name).is_some(),
                "nested array-item property `{name}` was dropped by sanitization"
            );
        }
        assert_eq!(item_props["title"]["type"], "string");
        assert_eq!(item_props["ref"]["description"], "local handle");
        assert_eq!(
            params["properties"]["issues"]["items"]["properties"]["verification_criteria"]["items"]
                ["type"],
            "string"
        );

        // Every `required` entry references a surviving property, so Gemini's
        // strict validator cannot 400 the request.
        let required = params["properties"]["issues"]["items"]["required"]
            .as_array()
            .expect("required array");
        for entry in required {
            let name = entry.as_str().expect("required entry is a string");
            assert!(
                item_props.get(name).is_some(),
                "required entry `{name}` has no matching property (would 400 Gemini)"
            );
        }

        // Annotation keywords are still stripped at the schema level.
        let serialized = serde_json::to_string(params).unwrap_or_default();
        assert!(!serialized.contains("additionalProperties"));
    }

    /// A property whose *name* collides with a stripped annotation keyword is
    /// preserved (the strip list only applies to schema keywords, not member
    /// names). Guards `additionalProperties` / `definitions` / `$defs` names too.
    #[test]
    fn test_property_named_like_stripped_keyword_survives() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["title", "additionalProperties", "definitions"],
            "properties": {
                "title": { "type": "string", "title": "annotation dropped" },
                "additionalProperties": { "type": "boolean" },
                "definitions": { "type": "string" }
            }
        });

        let sanitized = gemini_response_schema(&schema);
        let props = sanitized["properties"].as_object().expect("properties");
        assert!(props.contains_key("title"));
        assert!(props.contains_key("additionalProperties"));
        assert!(props.contains_key("definitions"));
        // The `title` *annotation* inside the `title` property subschema is
        // still stripped — only the member NAME is preserved.
        assert!(sanitized["properties"]["title"].get("title").is_none());
        assert_eq!(sanitized["properties"]["title"]["type"], "string");
    }

    /// The defensive post-pass drops a `required` entry that has no matching
    /// property, so a single dangling reference cannot 400 the whole request.
    #[test]
    fn test_prune_dangling_required_drops_unmatched_entry() {
        // `ghost` is required but never defined in `properties`.
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name", "ghost"],
            "properties": { "name": { "type": "string" } }
        });

        let sanitized = gemini_response_schema(&schema);
        let required: Vec<&str> = sanitized["required"]
            .as_array()
            .expect("required")
            .iter()
            .map(|v| v.as_str().unwrap_or_default())
            .collect();
        assert_eq!(required, vec!["name"]);
    }
}
