//! Agent SDK - A Rust SDK for building LLM-powered agents.
//!
//! This crate provides the building blocks for creating AI agents with:
//! - Tool execution and lifecycle hooks
//! - Streaming event-based architecture
//! - Provider-agnostic LLM interface
//! - Built-in primitive tools for file operations
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::{builder, ToolContext, ThreadId, providers::AnthropicProvider};
//!
//! // Build agent with defaults (in-memory stores, default hooks)
//! let agent = builder()
//!     .provider(AnthropicProvider::sonnet(api_key))
//!     .build();
//!
//! // Run the agent
//! let thread_id = ThreadId::new();
//! let tool_ctx = ToolContext::new(());
//! let mut events = agent.run(thread_id, "Hello!".to_string(), tool_ctx);
//!
//! while let Some(event) = events.recv().await {
//!     println!("{:?}", event);
//! }
//! ```
//!
//! # Custom Configuration
//!
//! ```ignore
//! use agent_sdk::{builder, AgentConfig, ToolRegistry, providers::AnthropicProvider};
//!
//! let agent = builder()
//!     .provider(AnthropicProvider::sonnet(api_key))
//!     .tools(my_tools)
//!     .config(AgentConfig {
//!         max_turns: 20,
//!         system_prompt: "You are a helpful assistant.".to_string(),
//!         ..Default::default()
//!     })
//!     .build();
//! ```
//!
//! # Custom Stores and Hooks
//!
//! ```ignore
//! use agent_sdk::builder;
//!
//! let agent = builder()
//!     .provider(my_provider)
//!     .hooks(my_hooks)
//!     .message_store(my_message_store)
//!     .state_store(my_state_store)
//!     .build_with_stores();
//! ```

#![forbid(unsafe_code)]

mod agent_loop;
mod capabilities;
mod environment;
mod events;
mod filesystem;
mod hooks;
pub mod llm;
pub mod primitive_tools;
pub mod providers;
pub mod skills;
mod stores;
mod tools;
mod types;

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
    AgentConfig, AgentState, PendingAction, ThreadId, TokenUsage, ToolResult, ToolTier,
};
