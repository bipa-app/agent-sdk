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
//! | [`audit`]       | Authoritative tool audit sink (full lifecycle outcomes) |
//! | [`stores`]      | Persistence traits for messages, state, events, and tool executions |
//! | [`environment`] | Filesystem / process environment abstraction |
//! | [`seed`]        | Durable reconstruction types (`ToolContextSeed`, `ExecutionContextFactory`) |

#![forbid(unsafe_code)]

pub mod audit;
pub mod authority;
pub mod environment;
pub mod hooks;
pub mod seed;
pub mod stores;
pub mod tools;

// Convenience re-exports
pub use audit::{NoopAuditSink, ToolAuditSink};
pub use authority::{EventAuthority, LocalEventAuthority};
pub use environment::{Environment, ExecResult, FileEntry, GrepMatch, NullEnvironment};
pub use hooks::{AgentHooks, AllowAllHooks, DefaultHooks, LoggingHooks, ToolDecision};
pub use seed::{DefaultContextFactory, ExecutionContextFactory, HostDependencies, ToolContextSeed};
pub use stores::{
    EventStore, InMemoryEventStore, InMemoryExecutionStore, InMemoryStore, MessageStore,
    StateStore, StoredTurnEvents, ToolExecutionStore,
};
pub use tools::{
    AsyncTool, DynamicToolName, ErasedAsyncTool, ErasedListenTool, ErasedTool, ErasedToolStatus,
    ListenExecuteTool, ListenStopReason, ListenToolUpdate, PrimitiveToolName, ProgressStage,
    SimpleTool, SimpleToolAdapter, Tool, ToolContext, ToolName, ToolRegistry, ToolStatus,
    TypedTool, TypedToolAdapter, invalid_tool_input_result, stage_to_string, tool_name_from_str,
    tool_name_to_string, validate_tool_input,
};

// Core audit record re-exports for downstream convenience.
pub use agent_sdk_foundation::audit::{
    AuditProvenance, ToolAuditOutcome, ToolAuditRecord, ToolAuditRecordParams,
};
