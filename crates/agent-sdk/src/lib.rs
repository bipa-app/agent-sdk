//! # Agent SDK
//!
//! A Rust SDK for building AI agents powered by large language models (LLMs).
//!
//! This crate provides the infrastructure to build agents that can:
//! - Converse with users via multiple LLM providers
//! - Execute tools to interact with external systems
//! - Persist turn events for downstream consumers and UIs
//! - Persist conversation history and state
//!
//! For task-oriented recipes — tools, typed tools, structured output,
//! streaming, MCP (local + remote HTTP), durable serving, and
//! human-in-the-loop — see the
//! [cookbook](https://github.com/bipa-app/agent-sdk/blob/main/crates/agent-sdk/COOKBOOK.md)
//! and the runnable
//! [`examples/`](https://github.com/bipa-app/agent-sdk/tree/main/crates/agent-sdk/examples).
//!
//! ## Quick Start
//!
//! Ask a question and print the answer — the whole 30-second path:
//!
//! ```no_run
//! use agent_sdk::{builder, ThreadId, providers::AnthropicProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::from_env()) // reads ANTHROPIC_API_KEY
//!     .build();
//!
//! let answer = agent.ask(ThreadId::new(), "What is the capital of France?").await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```
//!
//! [`ask`](AgentLoop::ask) builds the [`ToolContext`] and [`CancellationToken`]
//! internally and returns the assembled assistant text. When you need
//! application context, a confirmation flow, explicit cancellation, or the raw
//! [`AgentRunState`], drop down to [`run`](AgentLoop::run). The runnable example
//! below uses a tiny stub provider so it compiles **and runs** under
//! `cargo test --doc` with no network and no API key:
//!
//! ```
//! use std::sync::Arc;
//! use agent_sdk::{
//!     builder, AgentEvent, AgentInput, CancellationToken, EventStore, InMemoryEventStore,
//!     ThreadId, ToolContext,
//! };
//! use agent_sdk::llm::{
//!     ChatOutcome, ChatRequest, ChatResponse, ContentBlock, LlmProvider, StopReason, Usage,
//! };
//! use async_trait::async_trait;
//!
//! // A stub provider that always replies with a fixed line. Keeps the
//! // quickstart key-free and offline. Swap for `AnthropicProvider` (below)
//! // for real conversations.
//! struct StubProvider;
//!
//! #[async_trait]
//! impl LlmProvider for StubProvider {
//!     async fn chat(&self, _request: ChatRequest) -> anyhow::Result<ChatOutcome> {
//!         Ok(ChatOutcome::Success(ChatResponse {
//!             id: "stub".to_string(),
//!             content: vec![ContentBlock::Text { text: "Paris.".to_string() }],
//!             model: self.model().to_string(),
//!             stop_reason: Some(StopReason::EndTurn),
//!             usage: Usage {
//!                 input_tokens: 0,
//!                 output_tokens: 0,
//!                 cached_input_tokens: 0,
//!                 cache_creation_input_tokens: 0,
//!             },
//!         }))
//!     }
//!     fn model(&self) -> &str { "stub-model" }
//!     fn provider(&self) -> &'static str { "stub" }
//! }
//!
//! # async fn example() -> anyhow::Result<()> {
//! // 1. Build the agent.
//! let event_store = Arc::new(InMemoryEventStore::new());
//! let agent = builder::<()>()
//!     .provider(StubProvider)
//!     .event_store(event_store.clone())
//!     .build();
//!
//! // 2. Run a conversation.
//! let thread_id = ThreadId::new();
//! let final_state = agent.run(
//!     thread_id.clone(),
//!     AgentInput::Text("What is the capital of France?".to_string()),
//!     ToolContext::new(()),
//!     CancellationToken::new(),
//! );
//! let _ = final_state.await?;
//!
//! // 3. Read persisted events.
//! let mut reply = String::new();
//! for envelope in event_store.get_events(&thread_id).await? {
//!     match envelope.event {
//!         AgentEvent::Text { text, .. } => reply.push_str(&text),
//!         AgentEvent::Done { .. } => break,
//!         _ => {}
//!     }
//! }
//! assert_eq!(reply, "Paris.");
//! # Ok(())
//! # }
//! # tokio::runtime::Builder::new_current_thread()
//! #     .enable_all()
//! #     .build()
//! #     .unwrap()
//! #     .block_on(example())
//! #     .unwrap();
//! ```
//!
//! ### Run with a real key
//!
//! For a live conversation, depend on the `anthropic` provider and read your
//! key from the environment — the only change is the provider line:
//!
//! ```no_run
//! use agent_sdk::{builder, InMemoryEventStore, providers::AnthropicProvider};
//! use std::sync::Arc;
//!
//! # fn main() -> anyhow::Result<()> {
//! let api_key = std::env::var("ANTHROPIC_API_KEY")?;
//! let event_store = Arc::new(InMemoryEventStore::new());
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::sonnet(api_key))
//!     .event_store(event_store)
//!     .build();
//! # let _ = agent;
//! # Ok(())
//! # }
//! ```
//!
//! ## Core Concepts
//!
//! ### Agent Loop
//!
//! The [`AgentLoop`] orchestrates the conversation cycle:
//!
//! 1. User sends a message
//! 2. Agent sends message to LLM
//! 3. LLM responds with text and/or tool calls
//! 4. Agent executes tools and feeds results back to LLM
//! 5. Repeat until LLM responds with only text
//!
//! Use [`builder()`] to construct an agent:
//!
//! ```no_run
//! use agent_sdk::{builder, AgentConfig, providers::AnthropicProvider};
//!
//! # fn example() {
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::from_env())
//!     .config(AgentConfig {
//!         max_turns: Some(20),
//!         system_prompt: "You are a helpful assistant.".into(),
//!         ..Default::default()
//!     })
//!     .build();
//! # }
//! ```
//!
//! ### Tools
//!
//! Tools let the LLM interact with external systems. The lowest-ceremony way
//! is the [`SimpleTool`] trait — a `&'static str` name, no [`ToolName`] type:
//!
//! ```
//! use agent_sdk::{SimpleTool, ToolContext, ToolResult};
//! use serde_json::{json, Value};
//! use std::future::Future;
//!
//! struct WeatherTool;
//!
//! impl SimpleTool<()> for WeatherTool {
//!     fn name(&self) -> &'static str { "get_weather" }
//!     fn description(&self) -> &'static str { "Get current weather for a city" }
//!     fn input_schema(&self) -> Value {
//!         json!({ "type": "object", "properties": { "city": { "type": "string" } } })
//!     }
//!
//!     fn execute(
//!         &self,
//!         _ctx: &ToolContext<()>,
//!         input: Value,
//!     ) -> impl Future<Output = anyhow::Result<ToolResult>> + Send {
//!         async move {
//!             let city = input["city"].as_str().unwrap_or("Unknown");
//!             Ok(ToolResult::success(format!("Weather in {city}: Sunny, 72°F")))
//!         }
//!     }
//! }
//! ```
//!
//! Register it with [`ToolRegistry::register_simple`]:
//!
//! ```no_run
//! use agent_sdk::{builder, ToolRegistry, providers::AnthropicProvider};
//! # use agent_sdk::{SimpleTool, ToolContext, ToolResult};
//! # use serde_json::Value;
//! # use std::future::Future;
//! # struct WeatherTool;
//! # impl SimpleTool<()> for WeatherTool {
//! #     fn name(&self) -> &'static str { "get_weather" }
//! #     fn description(&self) -> &'static str { "" }
//! #     fn input_schema(&self) -> Value { Value::Null }
//! #     fn execute(&self, _: &ToolContext<()>, _: Value) -> impl Future<Output = anyhow::Result<ToolResult>> + Send {
//! #         async { Ok(ToolResult::success("")) }
//! #     }
//! # }
//!
//! # fn example() {
//! let mut tools = ToolRegistry::new();
//! tools.register_simple(WeatherTool);
//!
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::from_env())
//!     .tools(tools)
//!     .build();
//! # }
//! ```
//!
//! For full control over the serialized tool name, implement the [`Tool`]
//! trait directly with a strongly-typed [`ToolName`] (e.g. a `#[derive(Serialize,
//! Deserialize)]` enum or the built-in [`DynamicToolName`]).
//!
//! ### Tool Tiers
//!
//! Tools are classified by permission level via [`ToolTier`]:
//!
//! | Tier | Description | Example |
//! |------|-------------|---------|
//! | [`ToolTier::Observe`] | Read-only, always allowed | Get balance, read file |
//! | [`ToolTier::Confirm`] | Requires user confirmation | Send email, transfer funds |
//!
//! ### Lifecycle Hooks
//!
//! Implement [`AgentHooks`] to intercept and control agent behavior:
//!
//! ```
//! use agent_sdk::{AgentHooks, ToolDecision, ToolInvocation, ToolResult, ToolTier};
//! use async_trait::async_trait;
//!
//! struct MyHooks;
//!
//! #[async_trait]
//! impl AgentHooks for MyHooks {
//!     async fn pre_tool_use(&self, invocation: &ToolInvocation) -> ToolDecision {
//!         println!("Tool called: {}", invocation.tool_name);
//!         match invocation.tier {
//!             ToolTier::Observe => ToolDecision::Allow,
//!             ToolTier::Confirm => ToolDecision::RequiresConfirmation(
//!                 "Please confirm this action".into()
//!             ),
//!         }
//!     }
//!
//!     async fn post_tool_use(&self, tool_name: &str, result: &ToolResult) {
//!         println!("{tool_name} completed: {}", result.success);
//!     }
//! }
//! ```
//!
//! Built-in hook implementations:
//! - [`DefaultHooks`] - Tier-based permissions (default)
//! - [`AllowAllHooks`] - Allow all tools without confirmation (for testing)
//! - [`LoggingHooks`] - Debug logging for all events
//!
//! ### Events
//!
//! The agent emits [`AgentEvent`]s during execution for real-time updates:
//!
//! | Event | Description |
//! |-------|-------------|
//! | [`AgentEvent::Start`] | Agent begins processing |
//! | [`AgentEvent::Text`] | Text response from LLM |
//! | [`AgentEvent::TextDelta`] | Streaming text chunk |
//! | [`AgentEvent::ToolCallStart`] | Tool execution starting |
//! | [`AgentEvent::ToolCallEnd`] | Tool execution completed |
//! | [`AgentEvent::TurnComplete`] | One LLM round-trip finished |
//! | [`AgentEvent::Done`] | Agent completed successfully |
//! | [`AgentEvent::Error`] | An error occurred |
//!
//! ### Task Tracking
//!
//! Use [`TodoWriteTool`] and [`TodoReadTool`] to track task progress:
//!
//! ```no_run
//! use agent_sdk::todo::{TodoState, TodoWriteTool, TodoReadTool};
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//!
//! let state = Arc::new(RwLock::new(TodoState::new()));
//! let write_tool = TodoWriteTool::new(Arc::clone(&state));
//! let read_tool = TodoReadTool::new(state);
//! ```
//!
//! Task states: `Pending` (○), `InProgress` (⚡), `Completed` (✓)
//!
//! ### Custom Context
//!
//! Pass application-specific data to tools via the generic type parameter:
//!
//! ```
//! use agent_sdk::{DynamicToolName, Tool, ToolContext, ToolResult, ToolTier};
//! use serde_json::Value;
//! use std::future::Future;
//!
//! // Your application context
//! struct AppContext {
//!     user_id: String,
//!     // database: Database,
//! }
//!
//! struct UserInfoTool;
//!
//! impl Tool<AppContext> for UserInfoTool {
//!     type Name = DynamicToolName;
//!
//!     fn name(&self) -> DynamicToolName { DynamicToolName::new("get_user_info") }
//!     fn display_name(&self) -> &'static str { "User Info" }
//!     fn description(&self) -> &'static str { "Get info about current user" }
//!     fn input_schema(&self) -> Value { serde_json::json!({"type": "object"}) }
//!
//!     fn execute(
//!         &self,
//!         ctx: &ToolContext<AppContext>,
//!         _input: Value,
//!     ) -> impl Future<Output = anyhow::Result<ToolResult>> + Send {
//!         let user_id = ctx.app.user_id.clone();
//!         async move {
//!             Ok(ToolResult::success(format!("User: {user_id}")))
//!         }
//!     }
//! }
//! ```
//!
//! ## Workspace Architecture
//!
//! The SDK is split into focused crates. This `agent-sdk` crate is the
//! **public façade** — it re-exports everything you need so downstream
//! users only depend on `agent-sdk`.
//!
//! | Crate | Purpose |
//! |-------|---------|
//! | [`agent_sdk_core`] | Data-only contract types (IDs, events, LLM messages) |
//! | [`agent_sdk_tools`] | Tool traits, registry, hooks, stores, environment |
//! | [`agent_sdk_providers`] | LLM provider trait and first-party implementations |
//! | `agent-server` | Server-side orchestration (internal, not published) |
//! | **`agent-sdk`** | **This crate** — façade with agent loop, examples, and convenience re-exports |
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`providers`] | LLM provider implementations |
//! | [`primitive_tools`] | Built-in file operation tools (Read, Write, Edit, Glob, Grep, Bash) |
//! | [`llm`] | LLM abstraction layer |
//! | [`subagent`] | Nested agent execution with [`SubagentFactory`] |
//! | [`mcp`] | Model Context Protocol client (stdio + streamable-HTTP/SSE, resources/prompts) |
//! | [`todo`](mod@todo) | Task tracking tools ([`TodoWriteTool`], [`TodoReadTool`]) |
//! | [`user_interaction`] | User question/confirmation tools ([`AskUserQuestionTool`]) |
//! | [`web`] | Web search and fetch tools |
//! | [`skills`] | Custom skill/command loading |
//! | [`reminders`] | System reminder infrastructure for agent guidance |
//!
//! ## System Reminders
//!
//! The SDK includes a reminder system that provides contextual guidance to the AI agent
//! using the `<system-reminder>` XML tag pattern. Claude is trained to recognize these
//! tags and follow the instructions without mentioning them to users.
//!
//! ```
//! use agent_sdk::reminders::{wrap_reminder, ReminderConfig, ReminderTracker};
//!
//! // Wrap guidance in system-reminder tags
//! let reminder = wrap_reminder("Verify the output before proceeding.");
//!
//! // Configure reminder behavior
//! let config = ReminderConfig::new()
//!     .with_todo_reminder_turns(5)
//!     .with_repeated_action_threshold(3);
//! ```
//!
//! ## Feature Flags
//!
//! Providers and the heavier tool families are gated behind cargo features so
//! a minimal consumer only compiles (and only pulls the transitive
//! dependencies of) what it uses. The common case — an Anthropic agent — works
//! out of the box because `anthropic` is the only default feature.
//!
//! | Feature | Default | Pulls | Description |
//! |---------|---------|-------|-------------|
//! | `anthropic`    | **Yes** | — | Anthropic Messages API provider |
//! | `openai`       | No  | — | `OpenAI` Chat Completions + Responses providers |
//! | `openai-codex` | No  | `tokio-tungstenite` | `OpenAI` Codex / `ChatGPT` WebSocket provider |
//! | `gemini`       | No  | — | Google Gemini provider |
//! | `vertex`       | No  | — | Google Vertex AI provider (implies `anthropic` + `gemini`) |
//! | `cloudflare`   | No  | — | Cloudflare AI Gateway proxy (implies `anthropic` + `openai` + `gemini`) |
//! | `web`          | No  | `html2text` | [`web`] search + fetch tools |
//! | `mcp`          | No  | — | [`mcp`] Model Context Protocol client (stdio + streamable-HTTP/SSE) |
//! | `skills`       | No  | `serde_yaml_ng` | [`skills`] markdown skill loader |
//! | `otel`         | No  | `opentelemetry` | OpenTelemetry tracing instrumentation |
//!
//! A minimal Anthropic-only build pulls no WebSocket, HTML, or YAML crates:
//!
//! ```toml
//! agent-sdk = { version = "0.8", default-features = false, features = ["anthropic"] }
//! ```
//!
//! When `otel` is enabled, the SDK emits OpenTelemetry spans for agent
//! invocations, turns, LLM requests, tool execution, subagent runs, MCP
//! operations, and context compaction. See the `observability` module for details.

#![forbid(unsafe_code)]
// Enable the `doc(cfg(...))` feature-badge annotations on docs.rs (which
// builds with `--cfg docsrs` on nightly — see `[package.metadata.docs.rs]`).
// A regular stable build never sets `docsrs`, so this nightly-only feature
// flag is inert outside docs.rs.
#![cfg_attr(docsrs, feature(doc_cfg))]

// ── Private modules (owned by this crate) ────────────────────────────
mod agent_loop;
pub mod builtin_tools;
mod capabilities;
pub mod context;
mod filesystem;
pub mod primitive_tools;
pub mod reminders;
pub mod subagent;
pub mod todo;
pub mod user_interaction;

// ── Feature-gated tool modules (opt-in, pull extra deps) ──────────────
#[cfg(feature = "mcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
pub mod mcp;
#[cfg(feature = "skills")]
#[cfg_attr(docsrs, doc(cfg(feature = "skills")))]
pub mod skills;
#[cfg(feature = "web")]
#[cfg_attr(docsrs, doc(cfg(feature = "web")))]
pub mod web;

#[cfg(feature = "otel")]
#[cfg_attr(docsrs, doc(cfg(feature = "otel")))]
pub mod observability;

// ── Re-export modules from workspace crates ──────────────────────────
// These thin modules delegate to the extracted crates so that
// `use agent_sdk::llm::*` etc. keep working for downstream users.
mod authority;
mod environment;
mod events;
mod hooks;
pub mod llm;
pub mod model_capabilities;
pub mod providers;
mod seed;
mod stores;
mod tools;
mod types;

// ── Flat re-exports ──────────────────────────────────────────────────
// Grouped by source crate so the provenance is clear. The names kept at
// the crate root are the newcomer-facing surface; server/host contract
// types live under [`advanced`] so they don't dominate autocomplete or the
// docs.rs front page.

// agent-sdk (owned — agent loop)
pub use agent_loop::{
    AgentHandle, AgentLoop, AgentLoopBuilder, AgentLoopCompactionConfig, builder,
};
pub use capabilities::AgentCapabilities;
pub use filesystem::{InMemoryFileSystem, LocalFileSystem};
pub use tokio_util::sync::CancellationToken;

// agent-sdk-core (via thin modules)
pub use agent_sdk_core::privacy::{
    REDACTED_MARKER, RedactionLevel, RedactionPolicy, redact_error, redact_for_observability,
    redact_string, redact_value,
};
pub use events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
pub use types::{
    AgentConfig, AgentError, AgentInput, AgentRunState, AgentState, ExecutionStatus,
    ExternalToolResult, PendingToolCallInfo, RetryConfig, RunOptions, ThreadId, TokenUsage,
    ToolExecution, ToolInvocation, ToolOutcome, ToolResult, ToolRuntime, ToolTier, TurnOptions,
};

// agent-sdk-tools (via thin modules)
pub use environment::{Environment, ExecResult, FileEntry, GrepMatch, NullEnvironment};
pub use hooks::{
    AgentHooks, AllowAllHooks, DefaultHooks, LoggingHooks, NoopAuditSink, ToolAuditSink,
    ToolDecision,
};
pub use seed::{DefaultContextFactory, ToolContextSeed};
pub use stores::{
    EventStore, InMemoryEventStore, InMemoryExecutionStore, InMemoryStore, MessageStore,
    StateStore, StoredTurnEvents, ToolExecutionStore,
};
pub use tools::{
    AsyncTool, DynamicToolName, PrimitiveToolName, ProgressStage, SimpleTool, SimpleToolAdapter,
    Tool, ToolContext, ToolLogic, ToolName, ToolRegistry, ToolStatus, TypedTool, TypedToolAdapter,
    invalid_tool_input_result, stage_to_string, tool_name_from_str, tool_name_to_string,
    validate_tool_input,
};

// ── Ergonomics macros (Phase 13·E) ───────────────────────────────────
// `#[derive(Tool)]`, `#[derive(TypedTool)]`, and `#[derive(ToolName)]` are
// re-exported from the proc-macro crate so a user writes
// `use agent_sdk::Tool;` for both the trait and its derive (the trait/derive
// share a name, which is idiomatic — cf. `serde::Serialize`). The generated
// code refers to `::agent_sdk::…` paths and the `__macro_support` module
// below, so consumers never depend on `agent-sdk-macros` directly. The
// declarative `tool!` macro is defined in this crate (see the macro
// definition further down).
#[doc(inline)]
pub use agent_sdk_macros::{Tool, ToolName, TypedTool};

/// Re-exports the macros need at their expansion site. **Not** part of the
/// stable public API — do not depend on it directly. It exists only so the
/// `#[derive(...)]` output can name `serde_json`/`serde`/`anyhow`/`schemars`
/// items through a single `::agent_sdk::__macro_support::…` path, regardless of
/// which of those crates the consumer has in scope.
#[doc(hidden)]
pub mod __macro_support {
    pub use anyhow::Result;
    pub use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub use serde_json::{Value, json, to_value};
    // Re-export `serde` itself so the generated `#[serde(crate = "...")]`
    // attribute on the `ToolName` mirror enum resolves without the consumer
    // having `serde` in their dependency tree under that exact name.
    pub use serde;

    /// `schemars::schema_for` shim for `#[tool(schema = "derive")]`.
    ///
    /// Only present under the `macros-schema` feature; the macro emits a
    /// `compile_error!` (not a missing-path error) when the feature is off, so
    /// this absence is never the diagnostic a user sees.
    #[cfg(feature = "macros-schema")]
    #[must_use]
    pub fn schema_for<T: schemars::JsonSchema>() -> schemars::Schema {
        schemars::schema_for!(T)
    }
}

/// Define a tool inline, expanding to a fresh zero-sized struct plus a
/// [`SimpleTool`] impl — the lowest-ceremony way to add a one-off tool in an
/// example, test, or script.
///
/// This is the declarative counterpart to [`derive@Tool`]: use the derive when
/// you want a named, reusable tool type; reach for `tool!` when you just need a
/// closure-like tool right where you register it.
///
/// The application context type defaults to `()`; pass `context: MyCtx,` before
/// the closure to use a different one. The closure receives
/// `&ToolContext<Ctx>` and the raw `serde_json::Value` arguments and must
/// return a future resolving to `anyhow::Result<ToolResult>`.
///
/// # Example
///
/// ```
/// use agent_sdk::{tool, ToolResult, ToolRegistry};
/// use serde_json::json;
///
/// let weather = tool! {
///     name: "get_weather",
///     description: "Get the current weather for a city",
///     schema: json!({
///         "type": "object",
///         "properties": { "city": { "type": "string" } },
///         "required": ["city"],
///     }),
///     |_ctx, input| async move {
///         let city = input["city"].as_str().unwrap_or("Unknown");
///         Ok(ToolResult::success(format!("Weather in {city}: Sunny")))
///     }
/// };
///
/// let mut registry: ToolRegistry<()> = ToolRegistry::new();
/// registry.register_simple(weather);
/// ```
#[macro_export]
macro_rules! tool {
    // Default context = ().
    (
        name: $name:expr,
        description: $description:expr,
        schema: $schema:expr,
        | $ctx:ident , $input:ident | $body:expr $(,)?
    ) => {
        $crate::tool! {
            name: $name,
            description: $description,
            schema: $schema,
            context: (),
            |$ctx, $input| $body
        }
    };
    // Explicit context type.
    (
        name: $name:expr,
        description: $description:expr,
        schema: $schema:expr,
        context: $ctxty:ty,
        | $ctx:ident , $input:ident | $body:expr $(,)?
    ) => {{
        struct __InlineTool;

        impl $crate::SimpleTool<$ctxty> for __InlineTool {
            fn name(&self) -> &'static str {
                $name
            }

            fn description(&self) -> &'static str {
                $description
            }

            fn input_schema(&self) -> $crate::__macro_support::Value {
                $schema
            }

            fn execute(
                &self,
                $ctx: &$crate::ToolContext<$ctxty>,
                $input: $crate::__macro_support::Value,
            ) -> impl ::core::future::Future<
                Output = $crate::__macro_support::Result<$crate::ToolResult>,
            > + ::core::marker::Send {
                $body
            }
        }

        __InlineTool
    }};
}

// agent-sdk-providers (via thin modules)
pub use llm::{ContentBlock, ContentSource, Effort, LlmProvider, ThinkingConfig, ThinkingMode};
pub use model_capabilities::{
    ModelCapabilities, PricePoint, Pricing, SourceStatus, get_model_capabilities,
    supported_model_capabilities,
};

// Schema-validated structured output (Phase 13): the [`ResponseFormat`] request
// field, the bounded re-prompt runner, and its typed result/error.
pub use agent_sdk_core::llm::ResponseFormat;
pub use agent_sdk_providers::{
    StructuredConfig, StructuredOutput, StructuredOutputError, StructuredOutputSupport,
    run_structured,
};

// ── Advanced / server-internal contract types ───────────────────────
/// Server- and host-facing contract types.
///
/// These types are not needed by a typical in-process agent. They form the
/// authoritative boundary that `agent-server` / `agent-service-host` build
/// on: per-turn outcome and summary contracts, the durable continuation
/// envelope, the audit-record protocol, the listen/erased tool plumbing, and
/// the worker-context reconstruction factory.
///
/// They are grouped here so that `agent_sdk::` autocomplete and the docs.rs
/// front page stay dominated by the newcomer-facing surface (see
/// [`prelude`]). Everything here remains a stable, public re-export — moving
/// a name into `advanced` is a path change, not a removal.
pub mod advanced {
    // Per-turn outcome / summary contract and the durable continuation.
    pub use crate::types::{
        AgentContinuation, CONTINUATION_VERSION, ContinuationEnvelope, ListenExecutionContext,
        TurnOutcome, TurnSummary,
    };

    // Audit-record protocol emitted at every tool-lifecycle transition.
    pub use agent_sdk_core::audit::{
        AuditProvenance, ToolAuditOutcome, ToolAuditRecord, ToolAuditRecordParams,
    };

    // Event-sequencing authority used by the server commit path.
    pub use crate::authority::{EventAuthority, LocalEventAuthority};

    // Worker-context reconstruction for externalized tool runtimes.
    pub use crate::seed::{ExecutionContextFactory, HostDependencies};

    // Listen/execute tool protocol and the type-erased registry wrappers.
    pub use crate::tools::{
        ErasedAsyncTool, ErasedListenTool, ErasedTool, ErasedToolStatus, ListenExecuteTool,
        ListenStopReason, ListenToolUpdate,
    };
}

// ── Prelude ──────────────────────────────────────────────────────────
/// The common imports for building an in-process agent.
///
/// `use agent_sdk::prelude::*;` brings the ~dozen names a newcomer needs:
/// the [`builder`], configuration and I/O types, the [`Tool`] surface, the
/// in-memory event store, the cancellation token, and — when the `anthropic`
/// feature is enabled (the default) — the
/// [`AnthropicProvider`](crate::providers::AnthropicProvider). Server-only
/// contract types are intentionally excluded — reach for [`crate::advanced`]
/// when you need them.
pub mod prelude {
    pub use crate::builder;
    #[cfg(feature = "anthropic")]
    pub use crate::providers::AnthropicProvider;
    // `tool!` (declarative) plus the `Tool` / `TypedTool` / `ToolName` derives
    // ride along on the same-named trait re-exports below — importing the trait
    // also imports the derive macro (cf. `serde::Serialize`).
    pub use crate::tool;
    pub use crate::{
        AgentConfig, AgentEvent, AgentInput, CancellationToken, DynamicToolName,
        InMemoryEventStore, SimpleTool, Tool, ToolContext, ToolName, ToolRegistry, ToolResult,
        ToolTier, TypedTool,
    };
}

// Convenience re-exports
pub use reminders::{
    ReminderConfig, ReminderTracker, ReminderTrigger, ToolReminder, append_reminder, wrap_reminder,
};
pub use subagent::{
    METADATA_MAX_SUBAGENT_DEPTH, METADATA_SUBAGENT_DEPTH, SubagentConfig, SubagentFactory,
    SubagentTool,
};
pub use todo::{TodoItem, TodoReadTool, TodoState, TodoStatus, TodoWriteTool};
pub use user_interaction::{
    AskUserQuestionTool, ConfirmationRequest, ConfirmationResponse, QuestionOption,
    QuestionRequest, QuestionResponse,
};

#[cfg(feature = "otel")]
#[cfg_attr(docsrs, doc(cfg(feature = "otel")))]
pub use observability::{
    CaptureDecision, CaptureKind, CaptureResult, ObservabilityStore, PayloadBundle,
};
