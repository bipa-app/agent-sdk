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
//! use agent_sdk::{
//!     AgentLoop, AgentConfig, InMemoryStore, DefaultHooks, ToolRegistry, ToolContext, ThreadId,
//!     providers::AnthropicProvider,
//! };
//!
//! let provider = AnthropicProvider::sonnet(api_key);
//! let tools = ToolRegistry::new();
//! let hooks = DefaultHooks;
//! let message_store = InMemoryStore::new();
//! let state_store = InMemoryStore::new();
//! let config = AgentConfig::default();
//!
//! let agent = AgentLoop::new(provider, tools, hooks, message_store, state_store, config);
//!
//! let thread_id = ThreadId::new();
//! let tool_ctx = ToolContext::new(());
//! let mut events = agent.run(thread_id, "Hello!".to_string(), tool_ctx);
//!
//! while let Some(event) = events.recv().await {
//!     println!("{:?}", event);
//! }
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
mod stores;
mod tools;
mod types;

pub use agent_loop::AgentLoop;
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
