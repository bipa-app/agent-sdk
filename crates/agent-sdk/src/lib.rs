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
//! ## Quick Start
//!
//! ```no_run
//! use agent_sdk::{
//!     builder, AgentEvent, AgentInput, CancellationToken, EventStore, InMemoryEventStore,
//!     ThreadId, ToolContext, providers::AnthropicProvider,
//! };
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // 1. Create an LLM provider
//! let api_key = std::env::var("ANTHROPIC_API_KEY")?;
//! let provider = AnthropicProvider::sonnet(api_key);
//!
//! // 2. Build the agent
//! let event_store = Arc::new(InMemoryEventStore::new());
//! let agent = builder::<()>()
//!     .provider(provider)
//!     .event_store(event_store.clone())
//!     .build();
//!
//! // 3. Run a conversation
//! let thread_id = ThreadId::new();
//! let ctx = ToolContext::new(());
//! let cancel = CancellationToken::new();
//! let final_state = agent.run(
//!     thread_id.clone(),
//!     AgentInput::Text("Hello!".to_string()),
//!     ctx,
//!     cancel,
//! );
//! let _ = final_state.await?;
//!
//! // 4. Read persisted events
//! for envelope in event_store.get_events(&thread_id).await? {
//!     match envelope.event {
//!         AgentEvent::Text { message_id: _, text } => print!("{text}"),
//!         AgentEvent::Done { .. } => break,
//!         _ => {}
//!     }
//! }
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
//! use agent_sdk::{builder, AgentConfig, InMemoryEventStore, providers::AnthropicProvider};
//! use std::sync::Arc;
//!
//! # fn example() {
//! # let api_key = String::new();
//! let event_store = Arc::new(InMemoryEventStore::new());
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::sonnet(api_key))
//!     .config(AgentConfig {
//!         max_turns: Some(20),
//!         system_prompt: "You are a helpful assistant.".into(),
//!         ..Default::default()
//!     })
//!     .event_store(event_store)
//!     .build();
//! # }
//! ```
//!
//! ### Tools
//!
//! Tools let the LLM interact with external systems. Implement the [`Tool`] trait:
//!
//! ```
//! use agent_sdk::{DynamicToolName, Tool, ToolContext, ToolResult, ToolTier};
//! use serde_json::{json, Value};
//! use std::future::Future;
//!
//! struct WeatherTool;
//!
//! // No #[async_trait] needed - Rust 1.75+ supports native async traits
//! impl Tool<()> for WeatherTool {
//!     type Name = DynamicToolName;
//!
//!     fn name(&self) -> DynamicToolName { DynamicToolName::new("get_weather") }
//!
//!     fn display_name(&self) -> &'static str { "Weather" }
//!
//!     fn description(&self) -> &'static str {
//!         "Get current weather for a city"
//!     }
//!
//!     fn input_schema(&self) -> Value {
//!         json!({
//!             "type": "object",
//!             "properties": {
//!                 "city": { "type": "string" }
//!             },
//!             "required": ["city"]
//!         })
//!     }
//!
//!     fn tier(&self) -> ToolTier { ToolTier::Observe }
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
//! Register tools with [`ToolRegistry`]:
//!
//! ```no_run
//! use agent_sdk::{builder, DynamicToolName, ToolRegistry, providers::AnthropicProvider};
//! # use agent_sdk::{Tool, ToolContext, ToolResult, ToolTier};
//! # use serde_json::Value;
//! # use std::future::Future;
//! # struct WeatherTool;
//! # impl Tool<()> for WeatherTool {
//! #     type Name = DynamicToolName;
//! #     fn name(&self) -> DynamicToolName { DynamicToolName::new("weather") }
//! #     fn display_name(&self) -> &'static str { "" }
//! #     fn description(&self) -> &'static str { "" }
//! #     fn input_schema(&self) -> Value { Value::Null }
//! #     fn execute(&self, _: &ToolContext<()>, _: Value) -> impl Future<Output = anyhow::Result<ToolResult>> + Send {
//! #         async { Ok(ToolResult::success("")) }
//! #     }
//! # }
//!
//! # fn example() {
//! # let api_key = String::new();
//! let mut tools = ToolRegistry::new();
//! tools.register(WeatherTool);
//!
//! let event_store = std::sync::Arc::new(agent_sdk::InMemoryEventStore::new());
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::sonnet(api_key))
//!     .tools(tools)
//!     .event_store(event_store)
//!     .build();
//! # }
//! ```
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
//! | [`mcp`] | Model Context Protocol support |
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
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `otel`  | No      | Enable OpenTelemetry tracing instrumentation |
//!
//! When `otel` is enabled, the SDK emits OpenTelemetry spans for agent
//! invocations, turns, LLM requests, tool execution, subagent runs, MCP
//! operations, and context compaction. See the `observability` module for details.

#![forbid(unsafe_code)]

// ── Private modules (owned by this crate) ────────────────────────────
mod agent_loop;
mod capabilities;
pub mod context;
mod filesystem;
pub mod mcp;
pub mod primitive_tools;
pub mod reminders;
pub mod skills;
pub mod subagent;
pub mod todo;
pub mod user_interaction;
pub mod web;

#[cfg(feature = "otel")]
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
// Grouped by source crate so the provenance is clear.

// agent-sdk (owned — agent loop)
pub use agent_loop::{
    AgentHandle, AgentLoop, AgentLoopBuilder, AgentLoopCompactionConfig, builder,
};
pub use capabilities::AgentCapabilities;
pub use filesystem::{InMemoryFileSystem, LocalFileSystem};
pub use tokio_util::sync::CancellationToken;

// agent-sdk-core (via thin modules)
pub use authority::{EventAuthority, LocalEventAuthority};
pub use events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
pub use types::{
    AgentConfig, AgentContinuation, AgentError, AgentInput, AgentRunState, AgentState,
    CONTINUATION_VERSION, ContinuationEnvelope, ExecutionStatus, ExternalToolResult,
    ListenExecutionContext, PendingToolCallInfo, RetryConfig, ThreadId, TokenUsage, ToolExecution,
    ToolInvocation, ToolOutcome, ToolResult, ToolRuntime, ToolTier, TurnOptions, TurnOutcome,
};

// agent-sdk-tools (via thin modules)
pub use environment::{Environment, ExecResult, FileEntry, GrepMatch, NullEnvironment};
pub use hooks::{AgentHooks, AllowAllHooks, DefaultHooks, LoggingHooks, ToolDecision};
pub use seed::{DefaultContextFactory, ExecutionContextFactory, HostDependencies, ToolContextSeed};
pub use stores::{
    EventStore, InMemoryEventStore, InMemoryExecutionStore, InMemoryStore, MessageStore,
    StateStore, StoredTurnEvents, ToolExecutionStore,
};
pub use tools::{
    AsyncTool, DynamicToolName, ErasedAsyncTool, ErasedListenTool, ErasedTool, ErasedToolStatus,
    ListenExecuteTool, ListenStopReason, ListenToolUpdate, PrimitiveToolName, ProgressStage, Tool,
    ToolContext, ToolName, ToolRegistry, ToolStatus, stage_to_string, tool_name_from_str,
    tool_name_to_string,
};

// agent-sdk-providers (via thin modules)
pub use llm::{ContentBlock, ContentSource, Effort, LlmProvider, ThinkingConfig, ThinkingMode};
pub use model_capabilities::{
    ModelCapabilities, PricePoint, Pricing, SourceStatus, get_model_capabilities,
    supported_model_capabilities,
};

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
pub use observability::{
    CaptureDecision, CaptureKind, CaptureResult, ObservabilityStore, PayloadBundle,
};
