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
//!     builder, AgentEvent, ThreadId, ToolContext,
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
//! let mut events = agent.run(thread_id, "Hello!".into(), ctx);
//!
//! // 4. Process streaming events
//! while let Some(event) = events.recv().await {
//!     match event {
//!         AgentEvent::Text { text } => print!("{text}"),
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
//! use agent_sdk::{Tool, ToolContext, ToolResult, ToolTier};
//! use async_trait::async_trait;
//! use serde_json::{json, Value};
//!
//! struct WeatherTool;
//!
//! #[async_trait]
//! impl Tool<()> for WeatherTool {
//!     fn name(&self) -> &str { "get_weather" }
//!
//!     fn description(&self) -> &str {
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
//!     async fn execute(
//!         &self,
//!         _ctx: &ToolContext<()>,
//!         input: Value,
//!     ) -> anyhow::Result<ToolResult> {
//!         let city = input["city"].as_str().unwrap_or("Unknown");
//!         Ok(ToolResult::success(format!("Weather in {city}: Sunny, 72°F")))
//!     }
//! }
//! ```
//!
//! Register tools with [`ToolRegistry`]:
//!
//! ```no_run
//! use agent_sdk::{builder, ToolRegistry, providers::AnthropicProvider};
//! # use agent_sdk::{Tool, ToolContext, ToolResult, ToolTier};
//! # use async_trait::async_trait;
//! # use serde_json::Value;
//! # struct WeatherTool;
//! # #[async_trait]
//! # impl Tool<()> for WeatherTool {
//! #     fn name(&self) -> &str { "weather" }
//! #     fn description(&self) -> &str { "" }
//! #     fn input_schema(&self) -> Value { Value::Null }
//! #     async fn execute(&self, _: &ToolContext<()>, _: Value) -> anyhow::Result<ToolResult> {
//! #         Ok(ToolResult::success(""))
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
//! | [`ToolTier::Confirm`] | Requires user confirmation | Send email, write file |
//! | [`ToolTier::RequiresPin`] | Requires PIN verification | Transfer funds |
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
//!             ToolTier::RequiresPin => ToolDecision::RequiresPin(
//!                 "Enter PIN to continue".into()
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
//! ### Custom Context
//!
//! Pass application-specific data to tools via the generic type parameter:
//!
//! ```
//! use agent_sdk::{Tool, ToolContext, ToolResult, ToolTier};
//! use async_trait::async_trait;
//! use serde_json::Value;
//!
//! // Your application context
//! struct AppContext {
//!     user_id: String,
//!     // database: Database,
//! }
//!
//! struct UserInfoTool;
//!
//! #[async_trait]
//! impl Tool<AppContext> for UserInfoTool {
//!     fn name(&self) -> &str { "get_user_info" }
//!     fn description(&self) -> &str { "Get info about current user" }
//!     fn input_schema(&self) -> Value { serde_json::json!({"type": "object"}) }
//!
//!     async fn execute(
//!         &self,
//!         ctx: &ToolContext<AppContext>,
//!         _input: Value,
//!     ) -> anyhow::Result<ToolResult> {
//!         // Access your context
//!         let user_id = &ctx.app.user_id;
//!         Ok(ToolResult::success(format!("User: {user_id}")))
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
//! | [`subagent`] | Nested agent execution |
//! | [`mcp`] | Model Context Protocol support |
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
pub use events::AgentEvent;
pub use filesystem::{InMemoryFileSystem, LocalFileSystem};
pub use hooks::{AgentHooks, AllowAllHooks, DefaultHooks, LoggingHooks, ToolDecision};
pub use llm::LlmProvider;
pub use stores::{InMemoryStore, MessageStore, StateStore};
pub use tools::{Tool, ToolContext, ToolRegistry};
pub use types::{
    AgentConfig, AgentState, PendingAction, RetryConfig, ThreadId, TokenUsage, ToolResult, ToolTier,
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
