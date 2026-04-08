//! Tool surface contracts and infrastructure for the Agent SDK.
//!
//! This crate defines the traits and registries that tools implement and the
//! runtime dispatches against, without coupling to any specific LLM provider
//! or runtime loop implementation.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`tools`]       | Tool traits, registry, `ToolContext`, name types |
//! | [`hooks`]       | Lifecycle hooks (pre/post tool, events, errors) |
//! | [`stores`]      | Persistence traits for messages, state, events, and tool executions |
//! | [`environment`] | Filesystem / process environment abstraction |

#![forbid(unsafe_code)]

pub mod environment;
pub mod hooks;
pub mod stores;
pub mod tools;

// Convenience re-exports
pub use environment::{Environment, ExecResult, FileEntry, GrepMatch, NullEnvironment};
pub use hooks::{AgentHooks, AllowAllHooks, DefaultHooks, LoggingHooks, ToolDecision};
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
