//! LLM message and chat data types.
//!
//! These are the wire-format types shared between the runtime, providers,
//! and the server.  The module intentionally contains **no** async traits
//! or runtime-specific logic so it can be depended on from thin crates.

use serde::{Deserialize, Serialize};

// ── Thinking ──────────────────────────────────────────────────────────

/// The mode of extended thinking.
#[derive(Debug, Clone)]
pub enum ThinkingMode {
    /// Explicitly enabled with a token budget.
    Enabled { budget_tokens: u32 },
    /// Adaptive thinking — the model decides how much to think.
    Adaptive,
}

/// Effort level for adaptive thinking via `output_config`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
    Max,
}

/// Configuration for extended thinking.
///
/// When enabled, the model will show its reasoning process before
/// generating the final response.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    /// Which thinking mode to use.
    pub mode: ThinkingMode,
    /// Optional effort level (sent via `output_config`).
    pub effort: Option<Effort>,
}

impl ThinkingConfig {
    /// Default budget: 10,000 tokens.
    ///
    /// This provides enough capacity for meaningful reasoning on most tasks
    /// while keeping costs reasonable. Increase for complex multi-step problems.
    pub const DEFAULT_BUDGET_TOKENS: u32 = 10_000;

    /// Minimum budget required by the Anthropic API.
    pub const MIN_BUDGET_TOKENS: u32 = 1_024;

    /// Create a config with an explicit token budget (Enabled mode).
    #[must_use]
    pub const fn new(budget_tokens: u32) -> Self {
        Self {
            mode: ThinkingMode::Enabled { budget_tokens },
            effort: None,
        }
    }

    /// Create an adaptive thinking config.
    #[must_use]
    pub const fn adaptive() -> Self {
        Self {
            mode: ThinkingMode::Adaptive,
            effort: None,
        }
    }

    /// Create an adaptive thinking config with an effort level.
    #[must_use]
    pub const fn adaptive_with_effort(effort: Effort) -> Self {
        Self {
            mode: ThinkingMode::Adaptive,
            effort: Some(effort),
        }
    }

    /// Set the effort level on an existing config.
    #[must_use]
    pub const fn with_effort(mut self, effort: Effort) -> Self {
        self.effort = Some(effort);
        self
    }
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self::new(Self::DEFAULT_BUDGET_TOKENS)
    }
}

// ── Request / Response ────────────────────────────────────────────────

/// Controls whether the model must use a tool.
#[derive(Debug, Clone)]
pub enum ToolChoice {
    /// Let the model decide whether to use tools (default when `None`).
    Auto,
    /// Force the model to call a specific tool by name.
    Tool(String),
}

/// Requests that the model constrain its final answer to a JSON Schema.
///
/// This is the wire-level description of a structured-output request. The
/// runtime maps it to each provider's native capability:
///
/// - **`OpenAI` / Gemini**: native JSON-mode / structured-outputs
///   (`response_format` / `responseSchema`).
/// - **Anthropic**: tool-forcing fallback — the runtime injects a single
///   "respond" tool whose `input_schema` is [`schema`](Self::schema) and
///   forces the model to call it.
///
/// The runtime validates the model's final output against [`schema`](Self::schema)
/// and, on mismatch, bounded-re-prompts before failing with a typed error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// Stable identifier for the schema. Surfaced to providers that require a
    /// name (`OpenAI` `json_schema.name`, the Anthropic fallback tool name).
    pub name: String,
    /// The JSON Schema the final assistant output must satisfy.
    ///
    /// This is a raw JSON Schema document (an object), not a Rust type. Callers
    /// that derive schemas from Rust types can plug in `schemars` upstream and
    /// pass the resulting document here.
    pub schema: serde_json::Value,
    /// Whether the provider should enforce strict schema adherence when it
    /// supports a strict mode (`OpenAI` `strict: true`). Has no effect on
    /// providers without a strict mode.
    pub strict: bool,
}

impl ResponseFormat {
    /// Create a response format from a schema name and a JSON Schema document.
    ///
    /// Defaults to `strict = true` so providers with a strict mode enforce the
    /// schema rather than treating it as a hint.
    #[must_use]
    pub fn new(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            schema,
            strict: true,
        }
    }

    /// Set whether strict schema adherence is requested.
    #[must_use]
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
    pub max_tokens: u32,
    /// Whether `max_tokens` was explicitly configured by the caller.
    pub max_tokens_explicit: bool,
    /// Optional session identifier for provider-side prompt caching or routing.
    pub session_id: Option<String>,
    /// Optional provider-managed cached content reference.
    ///
    /// This currently maps to Gemini / Vertex AI `cachedContent` handles.
    pub cached_content: Option<String>,
    /// Optional extended thinking configuration.
    pub thinking: Option<ThinkingConfig>,
    /// Optional constraint on tool usage.
    ///
    /// When `None` the provider's default behaviour applies (typically `auto`).
    pub tool_choice: Option<ToolChoice>,
    /// Optional request for the final answer to be constrained to a JSON
    /// Schema.
    ///
    /// When `Some`, the provider maps this to its native JSON-mode /
    /// structured-output capability (or a tool-forcing fallback) and the
    /// runtime validates the final output against the schema. When `None`
    /// (default) the model responds freely.
    pub response_format: Option<ResponseFormat>,
}

impl ChatRequest {
    /// Default token budget used by [`ChatRequest::new`] when the caller does
    /// not set one explicitly. Providers clamp this to their own ceiling.
    pub const DEFAULT_MAX_TOKENS: u32 = 4096;

    /// Build a request from a system prompt and a message list, leaving every
    /// optional knob at its default.
    ///
    /// This is the ergonomic counterpart to the (still-public) struct literal:
    /// the common case only needs `system` + `messages`, so callers no longer
    /// have to spell out the eight `None`/default fields. Layer optional
    /// settings on with the chainable `with_*` setters:
    ///
    /// ```
    /// use agent_sdk_foundation::llm::{ChatRequest, Message, ToolChoice};
    ///
    /// let req = ChatRequest::new("You are helpful.", vec![Message::user("Hi")])
    ///     .with_max_tokens(1024)
    ///     .with_tool_choice(ToolChoice::Auto);
    /// ```
    #[must_use]
    pub fn new(system: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            system: system.into(),
            messages,
            tools: None,
            max_tokens: Self::DEFAULT_MAX_TOKENS,
            max_tokens_explicit: false,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
        }
    }

    /// Set the tool list the model may call.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the maximum output-token budget (marks it as explicitly configured).
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self.max_tokens_explicit = true;
        self
    }

    /// Set the session identifier (provider-side prompt caching / routing).
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the extended-thinking configuration.
    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Constrain tool usage (defaults to the provider's `auto` when unset).
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Request the final answer be constrained to the given JSON-Schema
    /// [`ResponseFormat`] (structured output).
    #[must_use]
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

impl Message {
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Content::Text(text.into()),
        }
    }

    #[must_use]
    pub const fn user_with_content(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: Content::Blocks(blocks),
        }
    }

    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Content::Text(text.into()),
        }
    }

    #[must_use]
    pub const fn assistant_with_content(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content: Content::Blocks(blocks),
        }
    }

    #[must_use]
    pub fn assistant_with_tool_use(
        text: Option<String>,
        id: impl Into<String>,
        name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        let mut blocks = Vec::new();
        if let Some(t) = text {
            blocks.push(ContentBlock::Text { text: t });
        }
        blocks.push(ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
            thought_signature: None,
        });
        Self {
            role: Role::Assistant,
            content: Content::Blocks(blocks),
        }
    }

    #[must_use]
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::User,
            content: Content::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error: if is_error { Some(true) } else { None },
            }]),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl Content {
    #[must_use]
    pub fn first_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            Self::Blocks(blocks) => blocks.iter().find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            }),
        }
    }
}

/// Source data for image and document content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSource {
    pub media_type: String,
    pub data: String,
}

impl ContentSource {
    #[must_use]
    pub fn new(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            data: data.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        /// Opaque signature for round-tripping thinking blocks back to the API.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        /// Gemini thought signature for preserving reasoning context.
        /// Required for Gemini 3 models when sending function calls back.
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    #[serde(rename = "image")]
    Image { source: ContentSource },

    #[serde(rename = "document")]
    Document { source: ContentSource },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    /// Human-readable display name shown in UI and audit records.
    pub display_name: String,
    /// Permission tier for this tool.
    pub tier: super::types::ToolTier,
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub id: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub usage: Usage,
}

impl ChatResponse {
    #[must_use]
    pub fn first_text(&self) -> Option<&str> {
        self.content.iter().find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
    }

    #[must_use]
    pub fn first_thinking(&self) -> Option<&str> {
        self.content.iter().find_map(|b| match b {
            ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        })
    }

    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, &serde_json::Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        })
    }

    #[must_use]
    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Refusal,
    ModelContextWindowExceeded,
    /// A stop reason this version of the SDK does not recognize.
    ///
    /// Providers may introduce new stop reasons at any time. Rather than
    /// failing deserialization of an otherwise-valid response (or a
    /// persisted/replayed audit row), unknown values map here via
    /// `#[serde(other)]`. Consumers should treat it like
    /// [`StopReason::EndTurn`] (turn finished, nothing actionable) unless
    /// they have a more specific fallback.
    #[serde(other)]
    Unknown,
}

impl StopReason {
    /// Stable discriminant string used for durable rows, metrics, and
    /// dashboards.  Matches the serde representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::ToolUse => "tool_use",
            Self::MaxTokens => "max_tokens",
            Self::StopSequence => "stop_sequence",
            Self::Refusal => "refusal",
            Self::ModelContextWindowExceeded => "model_context_window_exceeded",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    /// Total input tokens reported by the provider.
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Portion of `input_tokens` billed at a cached-input rate, when reported.
    #[serde(default)]
    pub cached_input_tokens: u32,
    /// Portion of `input_tokens` spent creating provider-side prompt cache entries.
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ChatOutcome {
    Success(ChatResponse),
    RateLimited,
    InvalidRequest(String),
    ServerError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_new_defaults_then_setters() {
        let req = ChatRequest::new("sys", vec![Message::user("hi")]);
        assert_eq!(req.system, "sys");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, ChatRequest::DEFAULT_MAX_TOKENS);
        assert!(!req.max_tokens_explicit);
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
        assert!(req.response_format.is_none());

        let req = req
            .with_max_tokens(1234)
            .with_tool_choice(ToolChoice::Auto)
            .with_response_format(ResponseFormat::new(
                "r",
                serde_json::json!({"type": "object"}),
            ))
            .with_session_id("s-1");
        assert_eq!(req.max_tokens, 1234);
        assert!(req.max_tokens_explicit);
        assert!(matches!(req.tool_choice, Some(ToolChoice::Auto)));
        assert!(req.response_format.is_some());
        assert_eq!(req.session_id.as_deref(), Some("s-1"));
    }

    #[test]
    fn stop_reason_known_values_round_trip() -> Result<(), serde_json::Error> {
        for (json, expected) in [
            ("\"end_turn\"", StopReason::EndTurn),
            ("\"tool_use\"", StopReason::ToolUse),
            ("\"max_tokens\"", StopReason::MaxTokens),
            ("\"stop_sequence\"", StopReason::StopSequence),
            ("\"refusal\"", StopReason::Refusal),
            (
                "\"model_context_window_exceeded\"",
                StopReason::ModelContextWindowExceeded,
            ),
        ] {
            let parsed: StopReason = serde_json::from_str(json)?;
            assert_eq!(parsed, expected);
            assert_eq!(serde_json::to_string(&parsed)?, json);
        }
        Ok(())
    }

    #[test]
    fn stop_reason_unknown_value_deserializes_to_unknown() -> Result<(), serde_json::Error> {
        // An unrecognized provider stop reason must not fail deserialization;
        // `#[serde(other)]` routes it to `StopReason::Unknown`.
        let parsed: StopReason = serde_json::from_str("\"some_future_reason\"")?;
        assert_eq!(parsed, StopReason::Unknown);
        assert_eq!(parsed.as_str(), "unknown");
        Ok(())
    }

    #[test]
    fn stop_reason_unknown_serializes_to_unknown() -> Result<(), serde_json::Error> {
        assert_eq!(serde_json::to_string(&StopReason::Unknown)?, "\"unknown\"");
        Ok(())
    }

    // ── ContentBlock wire format ────────────────────────────────
    //
    // `ContentBlock` is persisted durably (AgentContinuation.response_content,
    // AgentEvent::UserInput), so its tag strings and optional-field omission
    // are part of the wire contract. A tag rename or variant reorder must fail
    // a test here, not silently corrupt persisted threads.

    #[test]
    fn content_block_text_wire_format() -> Result<(), serde_json::Error> {
        let json = serde_json::to_value(ContentBlock::Text { text: "hi".into() })?;
        assert_eq!(json, serde_json::json!({"type": "text", "text": "hi"}));
        Ok(())
    }

    #[test]
    fn content_block_thinking_omits_none_signature() -> Result<(), serde_json::Error> {
        let none = serde_json::to_value(ContentBlock::Thinking {
            thinking: "t".into(),
            signature: None,
        })?;
        assert_eq!(
            none,
            serde_json::json!({"type": "thinking", "thinking": "t"})
        );

        let some = serde_json::to_value(ContentBlock::Thinking {
            thinking: "t".into(),
            signature: Some("sig".into()),
        })?;
        assert_eq!(
            some,
            serde_json::json!({"type": "thinking", "thinking": "t", "signature": "sig"})
        );
        Ok(())
    }

    #[test]
    fn content_block_tool_use_omits_none_thought_signature() -> Result<(), serde_json::Error> {
        let none = serde_json::to_value(ContentBlock::ToolUse {
            id: "i".into(),
            name: "n".into(),
            input: serde_json::json!({"a": 1}),
            thought_signature: None,
        })?;
        assert_eq!(
            none,
            serde_json::json!({"type": "tool_use", "id": "i", "name": "n", "input": {"a": 1}})
        );

        let some = serde_json::to_value(ContentBlock::ToolUse {
            id: "i".into(),
            name: "n".into(),
            input: serde_json::json!({}),
            thought_signature: Some("ts".into()),
        })?;
        assert_eq!(
            some.get("thought_signature").and_then(|v| v.as_str()),
            Some("ts")
        );
        Ok(())
    }

    #[test]
    fn content_block_tool_result_omits_none_is_error() -> Result<(), serde_json::Error> {
        let none = serde_json::to_value(ContentBlock::ToolResult {
            tool_use_id: "t".into(),
            content: "out".into(),
            is_error: None,
        })?;
        assert_eq!(
            none,
            serde_json::json!({"type": "tool_result", "tool_use_id": "t", "content": "out"})
        );

        let some = serde_json::to_value(ContentBlock::ToolResult {
            tool_use_id: "t".into(),
            content: "out".into(),
            is_error: Some(true),
        })?;
        assert_eq!(
            some.get("is_error").and_then(serde_json::Value::as_bool),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn content_block_remaining_variant_tags() -> Result<(), serde_json::Error> {
        assert_eq!(
            serde_json::to_value(ContentBlock::RedactedThinking { data: "d".into() })?,
            serde_json::json!({"type": "redacted_thinking", "data": "d"})
        );
        assert_eq!(
            serde_json::to_value(ContentBlock::Image {
                source: ContentSource::new("image/png", "b64"),
            })?,
            serde_json::json!({"type": "image", "source": {"media_type": "image/png", "data": "b64"}})
        );
        assert_eq!(
            serde_json::to_value(ContentBlock::Document {
                source: ContentSource::new("application/pdf", "b64"),
            })?,
            serde_json::json!({"type": "document", "source": {"media_type": "application/pdf", "data": "b64"}})
        );
        Ok(())
    }

    #[test]
    fn content_block_every_tag_round_trips() -> Result<(), serde_json::Error> {
        let blocks = vec![
            ContentBlock::Text { text: "t".into() },
            ContentBlock::Thinking {
                thinking: "th".into(),
                signature: Some("s".into()),
            },
            ContentBlock::RedactedThinking { data: "d".into() },
            ContentBlock::ToolUse {
                id: "i".into(),
                name: "n".into(),
                input: serde_json::json!({"x": 1}),
                thought_signature: None,
            },
            ContentBlock::ToolResult {
                tool_use_id: "t".into(),
                content: "c".into(),
                is_error: Some(true),
            },
            ContentBlock::Image {
                source: ContentSource::new("image/png", "b"),
            },
            ContentBlock::Document {
                source: ContentSource::new("application/pdf", "b"),
            },
        ];
        for block in blocks {
            let json = serde_json::to_value(&block)?;
            let back: ContentBlock = serde_json::from_value(json.clone())?;
            assert_eq!(serde_json::to_value(&back)?, json);
        }
        Ok(())
    }

    // ── Content (untagged) wire format ──────────────────────────

    #[test]
    fn content_text_serializes_as_bare_string() -> Result<(), serde_json::Error> {
        let json = serde_json::to_value(Content::Text("hello".into()))?;
        assert_eq!(json, serde_json::json!("hello"));
        let back: Content = serde_json::from_value(serde_json::json!("hello"))?;
        assert!(matches!(back, Content::Text(s) if s == "hello"));
        Ok(())
    }

    #[test]
    fn content_blocks_serialize_as_array_including_empty() -> Result<(), serde_json::Error> {
        let json = serde_json::to_value(Content::Blocks(vec![ContentBlock::Text {
            text: "x".into(),
        }]))?;
        assert_eq!(json, serde_json::json!([{"type": "text", "text": "x"}]));

        // Empty blocks → `[]` and must round-trip back to `Blocks`, not `Text`,
        // even though `Text` is the first untagged variant.
        let empty = serde_json::to_value(Content::Blocks(vec![]))?;
        assert_eq!(empty, serde_json::json!([]));
        let back: Content = serde_json::from_value(empty)?;
        assert!(matches!(back, Content::Blocks(b) if b.is_empty()));
        Ok(())
    }

    // ── Message wire format ─────────────────────────────────────

    #[test]
    fn message_wire_format_text_and_blocks() -> Result<(), serde_json::Error> {
        let user = serde_json::to_value(Message::user("hi"))?;
        assert_eq!(user, serde_json::json!({"role": "user", "content": "hi"}));

        let assistant =
            serde_json::to_value(Message::assistant_with_content(vec![ContentBlock::Text {
                text: "yo".into(),
            }]))?;
        assert_eq!(
            assistant,
            serde_json::json!({"role": "assistant", "content": [{"type": "text", "text": "yo"}]})
        );

        let back: Message =
            serde_json::from_value(serde_json::json!({"role": "user", "content": "hi"}))?;
        assert_eq!(back.role, Role::User);
        assert!(matches!(back.content, Content::Text(s) if s == "hi"));
        Ok(())
    }
}
