//! LLM message and chat data types.
//!
//! These are the wire-format types shared between the runtime, providers,
//! and the server.  The module intentionally contains **no** async traits
//! or runtime-specific logic so it can be depended on from thin crates.

use std::time::Duration;

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

/// Time-to-live for a provider-side prompt-cache breakpoint.
///
/// Only the values the Anthropic Messages API accepts are modelled, so the
/// enum maps losslessly onto the wire `ttl` string. Providers without an
/// equivalent control ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTtl {
    /// Five-minute ephemeral cache (the provider default).
    FiveMinutes,
    /// One-hour ephemeral cache (extended retention).
    OneHour,
}

impl CacheTtl {
    /// The wire string a provider sends for this TTL (`"5m"` / `"1h"`).
    #[must_use]
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::FiveMinutes => "5m",
            Self::OneHour => "1h",
        }
    }
}

/// Caller-facing control over provider-side prompt caching.
///
/// This is additive: a [`ChatRequest`] with `cache = None` preserves each
/// provider's default caching behaviour. Set it to shape (or disable) caching:
///
/// - `enabled = false` opts the request out of caching entirely — providers
///   send no `cache_control` breakpoints.
/// - `ttl` selects the cache retention window (Anthropic ephemeral TTL).
/// - `max_breakpoints` caps how many cache breakpoints the provider may emit,
///   in decreasing order of prefix stability (tools, then system, then the
///   conversation tail). `None` leaves the provider's default count.
///
/// Providers without a prompt-cache control ignore every field gracefully.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Whether prompt caching is enabled for this request.
    pub enabled: bool,
    /// Optional cache retention window. `None` uses the provider default.
    pub ttl: Option<CacheTtl>,
    /// Optional cap on the number of cache breakpoints the provider emits.
    pub max_breakpoints: Option<u8>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::enabled()
    }
}

impl CacheConfig {
    /// An enabled cache config with provider defaults (no TTL override, all
    /// breakpoints).
    #[must_use]
    pub const fn enabled() -> Self {
        Self {
            enabled: true,
            ttl: None,
            max_breakpoints: None,
        }
    }

    /// A config that opts the request out of provider-side caching.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            ttl: None,
            max_breakpoints: None,
        }
    }

    /// Set the cache retention window.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: CacheTtl) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Cap the number of cache breakpoints the provider may emit.
    #[must_use]
    pub const fn with_max_breakpoints(mut self, max_breakpoints: u8) -> Self {
        self.max_breakpoints = Some(max_breakpoints);
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
    /// Optional control over provider-side prompt caching.
    ///
    /// When `None` (default) each provider keeps its built-in caching
    /// behaviour. When `Some`, providers that support prompt caching honour
    /// the [`CacheConfig`] (TTL, opt-out, breakpoint cap); others ignore it.
    pub cache: Option<CacheConfig>,
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
            cache: None,
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

    /// Set the provider-side prompt-cache control ([`CacheConfig`]).
    #[must_use]
    pub const fn with_cache(mut self, cache: CacheConfig) -> Self {
        self.cache = Some(cache);
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

    /// Provider-owned reasoning state that must be replayed exactly on a
    /// later request, but must never be interpreted or surfaced by the SDK.
    ///
    /// `provider` names the wire protocol that owns `data`; providers must
    /// ignore blocks owned by a different protocol. The JSON payload is kept
    /// opaque so a provider can evolve its state-item shape without requiring
    /// another SDK wire-format change.
    #[serde(rename = "opaque_reasoning")]
    OpaqueReasoning {
        provider: String,
        data: serde_json::Value,
    },

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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// The provider rate-limited the request (HTTP 429).
    ///
    /// Carries the retry delay parsed from the response's `Retry-After`
    /// header when the provider supplied one (see [`parse_retry_after`]), so
    /// the caller can honour the server's hint instead of guessing a backoff.
    /// `None` when no usable `Retry-After` was present.
    RateLimited(Option<Duration>),
    InvalidRequest(String),
    ServerError(String),
}

/// Parse the value of an HTTP `Retry-After` header into a [`Duration`].
///
/// Per [RFC 9110 §10.2.3], `Retry-After` is either a non-negative number of
/// seconds (delta-seconds) or an IMF-fixdate HTTP timestamp
/// (`Sun, 06 Nov 1994 08:49:37 GMT`). For the date form the delay is the
/// difference between that instant and now; a timestamp at or before now (or
/// any value that cannot be parsed) yields `None`.
///
/// [RFC 9110 §10.2.3]: https://www.rfc-editor.org/rfc/rfc9110#section-10.2.3
#[must_use]
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    // delta-seconds: a bare non-negative integer number of seconds.
    if let Ok(seconds) = trimmed.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    // IMF-fixdate: compute the remaining delay from now, dropping past dates.
    let target = parse_imf_fixdate(trimmed)?;
    let now = time::OffsetDateTime::now_utc();
    if target <= now {
        return None;
    }
    (target - now).try_into().ok()
}

/// Parse an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`) as a UTC instant.
fn parse_imf_fixdate(value: &str) -> Option<time::OffsetDateTime> {
    // IMF-fixdate is always UTC ("GMT"); parse the civil datetime and assume
    // UTC. A custom description avoids depending on the `macros` feature.
    let format = time::format_description::parse_borrowed::<1>(
        "[weekday repr:short], [day] [month repr:short] [year] \
         [hour]:[minute]:[second] GMT",
    )
    .ok()?;
    time::PrimitiveDateTime::parse(value, &format)
        .ok()
        .map(time::PrimitiveDateTime::assume_utc)
}

// ─────────────────────────────────────────────────────────────────────
// Tool-use / tool-result balancing
// ─────────────────────────────────────────────────────────────────────

/// Default `tool_result` text used to close a `tool_use` block the user
/// cancelled (or otherwise abandoned) before it produced a real result.
///
/// Surfaced to the model so it understands the call did not run, rather
/// than silently dropping the loop. Used by [`balance_tool_results`].
pub const USER_CANCELLED_TOOL_RESULT: &str = "User cancelled";

/// Collect the `tool_use` block ids carried by a single message, in the
/// order they appear. Empty for any message that carries no `tool_use`
/// blocks (the common case for user messages and text-only assistant
/// turns).
fn message_tool_use_ids(message: &Message) -> Vec<&str> {
    match &message.content {
        Content::Text(_) => Vec::new(),
        Content::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect(),
    }
}

/// Collect the set of `tool_use_id`s answered by `tool_result` blocks in a
/// single message. Empty unless the message actually carries
/// `tool_result` blocks.
fn message_tool_result_ids(message: &Message) -> std::collections::HashSet<&str> {
    match &message.content {
        Content::Text(_) => std::collections::HashSet::new(),
        Content::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect(),
    }
}

/// Collect every `tool_use_id` answered by a `tool_result` block *anywhere*
/// in `messages`.
///
/// Answeredness is judged across the whole conversation, not just the
/// message immediately after a `tool_use`: an id that already has a real
/// `tool_result` somewhere must never be synthesized again, or balancing
/// would emit a duplicate `tool_result` for the same id (itself an API
/// rejection) and mislabel a successful call as cancelled.
fn all_answered_tool_use_ids(messages: &[Message]) -> std::collections::HashSet<&str> {
    messages.iter().flat_map(message_tool_result_ids).collect()
}

/// True when `messages` contains a `tool_use` block whose id is not
/// answered by any `tool_result` block anywhere in the conversation.
///
/// This is exactly the condition the Anthropic Messages API rejects with
/// *"`tool_use` ids were found without `tool_result` blocks immediately
/// after"*. It arises whenever a turn is interrupted after the assistant
/// `tool_use` was persisted but before every result was recorded — most
/// commonly when the user answers one of several questions and cancels
/// the rest, or cancels a tool mid-flight.
#[must_use]
pub fn has_unbalanced_tool_use(messages: &[Message]) -> bool {
    let answered = all_answered_tool_use_ids(messages);
    messages
        .iter()
        .flat_map(message_tool_use_ids)
        .any(|id| !answered.contains(id))
}

/// Close every unanswered `tool_use` loop in `messages`.
///
/// Re-balances the conversation so each `tool_use` block is answered by a
/// `tool_result` block in the immediately following message, synthesizing
/// an error `tool_result` carrying `cancel_text` for every id left
/// unanswered.
///
/// The Anthropic Messages API requires that an assistant message's
/// `tool_use` ids each have a matching `tool_result` in the *next*
/// message. A turn that is cancelled or abandoned after the assistant
/// `tool_use` was persisted — but before all tool results landed — leaves
/// the conversation unbalanced, and the next request 400s. This pass
/// closes those loops so the conversation can continue.
///
/// Behaviour per assistant `tool_use` message:
/// - An id that already has a real `tool_result` anywhere in the
///   conversation is left alone (never duplicated or relabelled cancelled).
/// - If the following message already answers some ids (the partial case:
///   the user answered one question and cancelled the others), the missing
///   results are appended to that existing message.
/// - Otherwise a fresh user message carrying the synthetic results is
///   inserted directly after the assistant message.
///
/// Idempotent and order-preserving: a no-op clone when history is already
/// balanced (see [`has_unbalanced_tool_use`]).
#[must_use]
pub fn balance_tool_results(messages: &[Message], cancel_text: &str) -> Vec<Message> {
    // Judge answeredness across the whole conversation so a real result
    // that is not at idx+1 still suppresses synthesis (no duplicate id).
    let answered = all_answered_tool_use_ids(messages);
    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + 1);
    let mut idx = 0;
    while idx < messages.len() {
        let message = &messages[idx];
        let tool_use_ids = message_tool_use_ids(message);
        if tool_use_ids.is_empty() {
            out.push(message.clone());
            idx += 1;
            continue;
        }

        let synthetic: Vec<ContentBlock> = tool_use_ids
            .iter()
            .filter(|id| !answered.contains(*id))
            .map(|id| ContentBlock::ToolResult {
                tool_use_id: (*id).to_owned(),
                content: cancel_text.to_owned(),
                is_error: Some(true),
            })
            .collect();

        out.push(message.clone());

        let next = messages.get(idx + 1);

        if synthetic.is_empty() {
            // Already balanced — leave the following message for the next
            // loop iteration to handle normally.
            idx += 1;
            continue;
        }

        // A following message that already carries tool_result blocks is
        // *the* results message for this turn (the partial-answer case):
        // merge the synthetic results into it. Anything else (a fresh user
        // prompt, another assistant turn, or end-of-history) gets a brand
        // new results message inserted right after the assistant turn.
        match next {
            Some(next_message) if !message_tool_result_ids(next_message).is_empty() => {
                let mut merged = next_message.clone();
                if let Content::Blocks(blocks) = &mut merged.content {
                    blocks.extend(synthetic);
                } else {
                    // A text-only message can't carry tool_result blocks, so
                    // this arm is unreachable given the guard above, but stay
                    // defensive rather than silently dropping the results.
                    merged.content = Content::Blocks(synthetic);
                }
                out.push(merged);
                idx += 2;
            }
            _ => {
                out.push(Message::user_with_content(synthetic));
                idx += 1;
            }
        }
    }
    out
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
        assert_eq!(
            serde_json::to_value(ContentBlock::OpaqueReasoning {
                provider: "test-provider".into(),
                data: serde_json::json!({"id": "reasoning_1", "encrypted": "ciphertext"}),
            })?,
            serde_json::json!({
                "type": "opaque_reasoning",
                "provider": "test-provider",
                "data": {"id": "reasoning_1", "encrypted": "ciphertext"}
            })
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
            ContentBlock::OpaqueReasoning {
                provider: "test-provider".into(),
                data: serde_json::json!({"id": "reasoning_1", "state": [1, 2, 3]}),
            },
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

    // ── Retry-After parsing ─────────────────────────────────────

    #[test]
    fn parse_retry_after_delta_seconds() {
        assert_eq!(parse_retry_after("125"), Some(Duration::from_secs(125)));
        assert_eq!(parse_retry_after("0"), Some(Duration::from_secs(0)));
        // Surrounding whitespace is tolerated.
        assert_eq!(parse_retry_after("  30 "), Some(Duration::from_secs(30)));
    }

    #[test]
    fn parse_retry_after_rejects_garbage_and_empty() {
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(parse_retry_after("   "), None);
        assert_eq!(parse_retry_after("soon"), None);
        // Negative deltas are not valid delta-seconds.
        assert_eq!(parse_retry_after("-5"), None);
    }

    #[test]
    fn parse_retry_after_past_imf_date_is_none() {
        // A date well in the past must not produce a (would-be negative) delay.
        assert_eq!(parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT"), None);
    }

    #[test]
    fn parse_retry_after_future_imf_date_is_some() {
        // Far-future date: must parse and yield a positive, large delay (the
        // 1_000_000s ≈ 11.6-day lower bound is trivially exceeded by a year-9999
        // target and avoids a round-unit literal).
        let parsed = parse_retry_after("Fri, 31 Dec 9999 23:59:59 GMT");
        assert!(parsed.is_some_and(|d| d > Duration::from_secs(1_000_000)));
    }

    // ── CacheConfig ─────────────────────────────────────────────

    #[test]
    fn cache_ttl_wire_strings() {
        assert_eq!(CacheTtl::FiveMinutes.as_wire_str(), "5m");
        assert_eq!(CacheTtl::OneHour.as_wire_str(), "1h");
    }

    #[test]
    fn cache_config_builders_and_default_request_cache_is_none() {
        let req = ChatRequest::new("sys", vec![Message::user("hi")]);
        assert!(
            req.cache.is_none(),
            "default request must not set a cache config"
        );

        let enabled = CacheConfig::enabled().with_ttl(CacheTtl::OneHour);
        assert!(enabled.enabled);
        assert_eq!(enabled.ttl, Some(CacheTtl::OneHour));
        assert_eq!(enabled.max_breakpoints, None);

        let disabled = CacheConfig::disabled();
        assert!(!disabled.enabled);

        let capped = CacheConfig::enabled().with_max_breakpoints(2);
        assert_eq!(capped.max_breakpoints, Some(2));

        let req = ChatRequest::new("s", vec![]).with_cache(CacheConfig::disabled());
        assert!(req.cache.is_some_and(|c| !c.enabled));
    }

    fn assistant_tool_uses(ids: &[&str]) -> Message {
        let blocks = ids
            .iter()
            .map(|id| ContentBlock::ToolUse {
                id: (*id).to_string(),
                name: "ask_user".to_string(),
                input: serde_json::json!({}),
                thought_signature: None,
            })
            .collect();
        Message::assistant_with_content(blocks)
    }

    fn tool_results(ids: &[&str]) -> Message {
        let blocks = ids
            .iter()
            .map(|id| ContentBlock::ToolResult {
                tool_use_id: (*id).to_string(),
                content: "answered".to_string(),
                is_error: None,
            })
            .collect();
        Message::user_with_content(blocks)
    }

    fn assert_balanced(messages: &[Message]) {
        assert!(
            !has_unbalanced_tool_use(messages),
            "expected balanced history, found an orphaned tool_use",
        );
    }

    #[test]
    fn balanced_history_is_left_untouched() {
        let messages = vec![
            Message::user("hi"),
            assistant_tool_uses(&["a"]),
            tool_results(&["a"]),
        ];
        assert!(!has_unbalanced_tool_use(&messages));
        let out = balance_tool_results(&messages, USER_CANCELLED_TOOL_RESULT);
        assert_eq!(out.len(), 3);
        assert_balanced(&out);
    }

    #[test]
    fn partial_cancellation_merges_into_existing_results_message() {
        // Four questions, one answered, three cancelled.
        let messages = vec![
            assistant_tool_uses(&["q1", "q2", "q3", "q4"]),
            tool_results(&["q1"]),
        ];
        assert!(has_unbalanced_tool_use(&messages));

        let out = balance_tool_results(&messages, USER_CANCELLED_TOOL_RESULT);
        assert_eq!(
            out.len(),
            2,
            "synthetic results merge into the existing message"
        );
        assert_balanced(&out);

        let Content::Blocks(blocks) = &out[1].content else {
            panic!("results message must carry blocks");
        };
        let cancelled: Vec<&str> = blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error: Some(true),
                } if content == USER_CANCELLED_TOOL_RESULT => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(cancelled, vec!["q2", "q3", "q4"]);
    }

    #[test]
    fn all_cancelled_with_no_following_message_appends_results() {
        // Cancel-all: the assistant turn is the last message, no results at all.
        let messages = vec![assistant_tool_uses(&["q1", "q2"])];
        assert!(has_unbalanced_tool_use(&messages));

        let out = balance_tool_results(&messages, USER_CANCELLED_TOOL_RESULT);
        assert_eq!(out.len(), 2, "a fresh results message is inserted");
        assert_eq!(out[1].role, Role::User);
        assert_balanced(&out);
    }

    #[test]
    fn orphan_followed_by_user_prompt_inserts_results_between() {
        // A fresh user turn arrived after an abandoned tool_use turn: the
        // results must be inserted *between* them, not after the prompt.
        let messages = vec![
            assistant_tool_uses(&["q1"]),
            Message::user("a brand new question from the user"),
        ];
        assert!(has_unbalanced_tool_use(&messages));

        let out = balance_tool_results(&messages, USER_CANCELLED_TOOL_RESULT);
        assert_eq!(out.len(), 3);
        assert_balanced(&out);
        // Order: assistant tool_use, synthetic results, then the user prompt.
        assert!(!message_tool_use_ids(&out[0]).is_empty());
        assert!(!message_tool_result_ids(&out[1]).is_empty());
        assert!(out[2].content.first_text() == Some("a brand new question from the user"));
    }

    #[test]
    fn balancing_is_idempotent() {
        let messages = vec![
            assistant_tool_uses(&["q1", "q2", "q3"]),
            tool_results(&["q2"]),
        ];
        let once = balance_tool_results(&messages, USER_CANCELLED_TOOL_RESULT);
        let twice = balance_tool_results(&once, USER_CANCELLED_TOOL_RESULT);
        assert_eq!(once.len(), twice.len());
        assert_balanced(&twice);
    }

    #[test]
    fn no_tool_use_history_is_a_noop() {
        let messages = vec![Message::user("hi"), Message::assistant("hello")];
        assert!(!has_unbalanced_tool_use(&messages));
        let out = balance_tool_results(&messages, USER_CANCELLED_TOOL_RESULT);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn real_result_not_at_idx1_is_not_duplicated_or_relabelled() {
        // A `tool_use` whose genuine result is separated from it by another
        // message must NOT get a synthetic "User cancelled" result — that
        // would emit two tool_result blocks for the same id (a 400) and lie
        // that a successful call was cancelled. Answeredness is judged over
        // the whole conversation, so the real result suppresses synthesis.
        let messages = vec![
            assistant_tool_uses(&["a"]),
            Message::user("an interjection between the call and its result"),
            tool_results(&["a"]),
        ];
        // No id is genuinely unanswered, so there is nothing to balance.
        assert!(!has_unbalanced_tool_use(&messages));

        let out = balance_tool_results(&messages, USER_CANCELLED_TOOL_RESULT);
        // Exactly one tool_result for "a", and none of them is a synthetic
        // cancellation.
        let a_results: Vec<&ContentBlock> = out
            .iter()
            .flat_map(|m| match &m.content {
                Content::Blocks(b) => b.as_slice(),
                Content::Text(_) => &[][..],
            })
            .filter(
                |b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "a"),
            )
            .collect();
        assert_eq!(a_results.len(), 1, "must not duplicate the real result");
        assert!(
            !matches!(a_results[0], ContentBlock::ToolResult { content, .. } if content == USER_CANCELLED_TOOL_RESULT),
            "the real successful result must not be relabelled cancelled",
        );
    }
}
