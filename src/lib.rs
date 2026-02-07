//! # Agent SDK
//!
//! A Rust SDK for building AI agents powered by large language models (LLMs).
//!
//! This crate provides the infrastructure to build agents that can:
//! - Converse with users via multiple LLM providers
//! - Execute tools to interact with external systems
//! - Stream events in real-time for responsive UIs
//! - Persist conversation history and state
//!
//! ## Quick Start
//!
//! ```no_run
//! use agent_sdk::{
//!     builder, AgentEvent, AgentInput, ThreadId, ToolContext,
//!     providers::AnthropicProvider,
//! };
//!
//! # async fn example() -> anyhow::Result<()> {
//! // 1. Create an LLM provider
//! let api_key = std::env::var("ANTHROPIC_API_KEY")?;
//! let provider = AnthropicProvider::sonnet(api_key);
//!
//! // 2. Build the agent
//! let agent = builder::<()>()
//!     .provider(provider)
//!     .build();
//!
//! // 3. Run a conversation
//! let thread_id = ThreadId::new();
//! let ctx = ToolContext::new(());
//! let (mut events, _final_state) = agent.run(
//!     thread_id,
//!     AgentInput::Text("Hello!".to_string()),
//!     ctx,
//! );
//!
//! // 4. Process streaming events
//! while let Some(envelope) = events.recv().await {
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
//! use agent_sdk::{builder, AgentConfig, providers::AnthropicProvider};
//!
//! # fn example() {
//! # let api_key = String::new();
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::sonnet(api_key))
//!     .config(AgentConfig {
//!         max_turns: 20,
//!         system_prompt: "You are a helpful assistant.".into(),
//!         ..Default::default()
//!     })
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
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::sonnet(api_key))
//!     .tools(tools)
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
//! use agent_sdk::{AgentHooks, ToolDecision, ToolResult, ToolTier};
//! use async_trait::async_trait;
//! use serde_json::Value;
//!
//! struct MyHooks;
//!
//! #[async_trait]
//! impl AgentHooks for MyHooks {
//!     async fn pre_tool_use(
//!         &self,
//!         tool_name: &str,
//!         _input: &Value,
//!         tier: ToolTier,
//!     ) -> ToolDecision {
//!         println!("Tool called: {tool_name}");
//!         match tier {
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
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`providers`] | LLM provider implementations |
//! | [`primitive_tools`] | Built-in file operation tools (Read, Write, Edit, Glob, Grep, Bash) |
//! | [`llm`] | LLM abstraction layer |
//! | [`subagent`] | Nested agent execution with [`SubagentFactory`] |
//! | [`mcp`] | Model Context Protocol support |
//! | [`todo`] | Task tracking tools ([`TodoWriteTool`], [`TodoReadTool`]) |
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
//! All features are enabled by default. The crate has no optional features currently.

#![forbid(unsafe_code)]

mod agent_loop;
mod capabilities;
pub mod context;
mod environment;
mod events;
mod filesystem;
mod hooks;
pub mod llm;
pub mod mcp;
pub mod primitive_tools;
pub mod providers;
pub mod reminders;
pub mod skills;
mod stores;
pub mod subagent;
pub mod todo;
mod tools;
mod types;
pub mod user_interaction;
pub mod web;

pub use agent_loop::{AgentLoop, AgentLoopBuilder, builder};
pub use capabilities::AgentCapabilities;
pub use environment::{Environment, ExecResult, FileEntry, GrepMatch, NullEnvironment};
pub use events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
pub use filesystem::{InMemoryFileSystem, LocalFileSystem};
pub use hooks::{AgentHooks, AllowAllHooks, DefaultHooks, LoggingHooks, ToolDecision};
pub use llm::{ContentBlock, ContentSource, LlmProvider, ThinkingConfig};
pub use stores::{
    InMemoryExecutionStore, InMemoryStore, MessageStore, StateStore, ToolExecutionStore,
};
pub use tools::{
    AsyncTool, DynamicToolName, ErasedAsyncTool, ErasedListenTool, ErasedTool, ErasedToolStatus,
    ListenExecuteTool, ListenStopReason, ListenToolUpdate, PrimitiveToolName, ProgressStage, Tool,
    ToolContext, ToolName, ToolRegistry, ToolStatus, stage_to_string, tool_name_from_str,
    tool_name_to_string,
};
pub use types::{
    AgentConfig, AgentContinuation, AgentError, AgentInput, AgentRunState, AgentState,
    ExecutionStatus, ListenExecutionContext, PendingToolCallInfo, RetryConfig, ThreadId,
    TokenUsage, ToolExecution, ToolOutcome, ToolResult, ToolTier, TurnOutcome,
};

// Re-export user interaction types for convenience
pub use user_interaction::{
    AskUserQuestionTool, ConfirmationRequest, ConfirmationResponse, QuestionOption,
    QuestionRequest, QuestionResponse,
};

// Re-export subagent types for convenience
pub use subagent::{SubagentConfig, SubagentFactory, SubagentTool};

// Re-export todo types for convenience
pub use todo::{TodoItem, TodoReadTool, TodoState, TodoStatus, TodoWriteTool};

// Re-export reminder types for convenience
pub use reminders::{
    ReminderConfig, ReminderTracker, ReminderTrigger, ToolReminder, append_reminder, wrap_reminder,
};
