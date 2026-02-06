//! Tool definition and registry.
//!
//! Tools allow the LLM to perform actions in the real world. This module provides:
//!
//! - [`Tool`] trait - Define custom tools the LLM can call
//! - [`ToolName`] trait - Marker trait for strongly-typed tool names
//! - [`PrimitiveToolName`] - Tool names for SDK's built-in tools
//! - [`DynamicToolName`] - Tool names created at runtime (MCP bridges)
//! - [`ToolRegistry`] - Collection of available tools
//! - [`ToolContext`] - Context passed to tool execution
//!
//! # Implementing a Tool
//!
//! ```ignore
//! use agent_sdk::{Tool, ToolContext, ToolResult, ToolTier, PrimitiveToolName};
//!
//! struct MyTool;
//!
//! // No #[async_trait] needed - Rust 1.75+ supports native async traits
//! impl Tool<MyContext> for MyTool {
//!     type Name = PrimitiveToolName;
//!
//!     fn name(&self) -> PrimitiveToolName { PrimitiveToolName::Read }
//!     fn display_name(&self) -> &'static str { "My Tool" }
//!     fn description(&self) -> &'static str { "Does something useful" }
//!     fn input_schema(&self) -> Value { json!({ "type": "object" }) }
//!     fn tier(&self) -> ToolTier { ToolTier::Observe }
//!
//!     async fn execute(&self, ctx: &ToolContext<MyContext>, input: Value) -> Result<ToolResult> {
//!         Ok(ToolResult::success("Done!"))
//!     }
//! }
//! ```

use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::llm;
use crate::types::{ToolOutcome, ToolResult, ToolTier};
use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;

// ============================================================================
// Tool Name Types
// ============================================================================

/// Marker trait for tool names.
///
/// Tool names must be serializable (for storage/logging) and deserializable
/// (for parsing from LLM responses). The string representation is derived
/// from serde serialization.
///
/// # Example
///
/// ```ignore
/// #[derive(Serialize, Deserialize)]
/// #[serde(rename_all = "snake_case")]
/// pub enum MyToolName {
///     Read,
///     Write,
/// }
///
/// impl ToolName for MyToolName {}
/// ```
pub trait ToolName: Send + Sync + Serialize + DeserializeOwned + 'static {}

/// Helper to get string representation of a tool name via serde.
///
/// # Panics
///
/// Panics if the tool name cannot be serialized to a string. This should
/// never happen with properly implemented `ToolName` types that use
/// `#[derive(Serialize)]`.
#[must_use]
pub fn tool_name_to_string<N: ToolName>(name: &N) -> String {
    serde_json::to_string(name)
        .expect("ToolName must serialize to string")
        .trim_matches('"')
        .to_string()
}

/// Parse a tool name from string via serde.
///
/// # Errors
/// Returns error if the string doesn't match a valid tool name.
pub fn tool_name_from_str<N: ToolName>(s: &str) -> Result<N, serde_json::Error> {
    serde_json::from_str(&format!("\"{s}\""))
}

/// Tool names for SDK's built-in primitive tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveToolName {
    Read,
    Write,
    Edit,
    MultiEdit,
    Bash,
    Glob,
    Grep,
    NotebookRead,
    NotebookEdit,
    TodoRead,
    TodoWrite,
    AskUser,
    LinkFetch,
    WebSearch,
}

impl ToolName for PrimitiveToolName {}

/// Dynamic tool name for runtime-created tools (MCP bridges, subagents).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DynamicToolName(String);

impl DynamicToolName {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl ToolName for DynamicToolName {}

// ============================================================================
// Progress Stage Types (for AsyncTool)
// ============================================================================

/// Marker trait for tool progress stages (type-safe, like [`ToolName`]).
///
/// Progress stages are used by async tools to indicate the current phase
/// of a long-running operation. They must be serializable for event streaming.
///
/// # Example
///
/// ```ignore
/// #[derive(Clone, Debug, Serialize, Deserialize)]
/// #[serde(rename_all = "snake_case")]
/// pub enum PixTransferStage {
///     Initiated,
///     Processing,
///     SentToBank,
/// }
///
/// impl ProgressStage for PixTransferStage {}
/// ```
pub trait ProgressStage: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {}

/// Helper to get string representation of a progress stage via serde.
///
/// # Panics
///
/// Panics if the stage cannot be serialized to a string. This should
/// never happen with properly implemented `ProgressStage` types.
#[must_use]
pub fn stage_to_string<S: ProgressStage>(stage: &S) -> String {
    serde_json::to_string(stage)
        .expect("ProgressStage must serialize to string")
        .trim_matches('"')
        .to_string()
}

/// Status update from an async tool operation.
#[derive(Clone, Debug, Serialize)]
pub enum ToolStatus<S: ProgressStage> {
    /// Operation is making progress
    Progress {
        stage: S,
        message: String,
        data: Option<serde_json::Value>,
    },

    /// Operation completed successfully
    Completed(ToolResult),

    /// Operation failed
    Failed(ToolResult),
}

/// Type-erased status for the agent loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ErasedToolStatus {
    /// Operation is making progress
    Progress {
        stage: String,
        message: String,
        data: Option<serde_json::Value>,
    },
    /// Operation completed successfully
    Completed(ToolResult),
    /// Operation failed
    Failed(ToolResult),
}

impl<S: ProgressStage> From<ToolStatus<S>> for ErasedToolStatus {
    fn from(status: ToolStatus<S>) -> Self {
        match status {
            ToolStatus::Progress {
                stage,
                message,
                data,
            } => Self::Progress {
                stage: stage_to_string(&stage),
                message,
                data,
            },
            ToolStatus::Completed(r) => Self::Completed(r),
            ToolStatus::Failed(r) => Self::Failed(r),
        }
    }
}

/// Context passed to tool execution
pub struct ToolContext<Ctx> {
    /// Application-specific context (e.g., `user_id`, db connection)
    pub app: Ctx,
    /// Tool-specific metadata
    pub metadata: HashMap<String, Value>,
    /// Optional channel for tools to emit events (e.g., subagent progress)
    event_tx: Option<mpsc::Sender<AgentEventEnvelope>>,
    /// Optional sequence counter for wrapping events in envelopes
    event_seq: Option<SequenceCounter>,
}

impl<Ctx> ToolContext<Ctx> {
    #[must_use]
    pub fn new(app: Ctx) -> Self {
        Self {
            app,
            metadata: HashMap::new(),
            event_tx: None,
            event_seq: None,
        }
    }

    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set the event channel and sequence counter for tools that need to emit
    /// events during execution.
    #[must_use]
    pub fn with_event_tx(
        mut self,
        tx: mpsc::Sender<AgentEventEnvelope>,
        seq: SequenceCounter,
    ) -> Self {
        self.event_tx = Some(tx);
        self.event_seq = Some(seq);
        self
    }

    /// Emit an event through the event channel (if set).
    ///
    /// The event is wrapped in an [`AgentEventEnvelope`] with a unique ID,
    /// sequence number, and timestamp before sending.
    ///
    /// This uses `try_send` to avoid blocking and to ensure the future is `Send`.
    /// The event is silently dropped if the channel is full.
    pub fn emit_event(&self, event: AgentEvent) {
        if let Some((tx, seq)) = self.event_tx.as_ref().zip(self.event_seq.as_ref()) {
            let envelope = AgentEventEnvelope::wrap(event, seq);
            let _ = tx.try_send(envelope);
        }
    }

    /// Get a clone of the event channel sender (if set).
    ///
    /// This is useful for tools that spawn subprocesses (like subagents)
    /// and need to forward events to the parent's event stream.
    #[must_use]
    pub fn event_tx(&self) -> Option<mpsc::Sender<AgentEventEnvelope>> {
        self.event_tx.clone()
    }

    /// Get a clone of the sequence counter (if set).
    ///
    /// This is useful for tools that spawn subprocesses (like subagents)
    /// and need to assign sequence numbers to events sent to the parent's stream.
    #[must_use]
    pub fn event_seq(&self) -> Option<SequenceCounter> {
        self.event_seq.clone()
    }
}

// ============================================================================
// Tool Trait
// ============================================================================

/// Definition of a tool that can be called by the agent.
///
/// Tools have a strongly-typed `Name` associated type that determines
/// how the tool name is serialized for LLM communication.
///
/// # Native Async Support
///
/// This trait uses Rust's native async functions in traits (stabilized in Rust 1.75).
/// You do NOT need the `async_trait` crate to implement this trait.
pub trait Tool<Ctx>: Send + Sync {
    /// The type of name for this tool.
    type Name: ToolName;

    /// Returns the tool's strongly-typed name.
    fn name(&self) -> Self::Name;

    /// Human-readable display name for UI (e.g., "Read File" vs "read").
    ///
    /// Defaults to empty string. Override for better UX.
    fn display_name(&self) -> &'static str;

    /// Human-readable description of what the tool does.
    fn description(&self) -> &'static str;

    /// JSON schema for the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Permission tier for this tool.
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    /// Execute the tool with the given input.
    ///
    /// # Errors
    /// Returns an error if tool execution fails.
    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Value,
    ) -> impl Future<Output = Result<ToolResult>> + Send;
}

// ============================================================================
// AsyncTool Trait
// ============================================================================

/// A tool that performs long-running async operations.
///
/// `AsyncTool`s have two phases:
/// 1. `execute()` - Start the operation (lightweight, returns quickly)
/// 2. `check_status()` - Stream progress until completion
///
/// The actual work should happen externally (background task, external service)
/// and persist results to a durable store. The tool is just an orchestrator.
///
/// # Example
///
/// ```ignore
/// impl AsyncTool<MyCtx> for ExecutePixTransferTool {
///     type Name = PixToolName;
///     type Stage = PixTransferStage;
///
///     async fn execute(&self, ctx: &ToolContext<MyCtx>, input: Value) -> Result<ToolOutcome> {
///         let params = parse_input(&input)?;
///         let operation_id = ctx.app.pix_service.start_transfer(params).await?;
///         Ok(ToolOutcome::in_progress(
///             operation_id,
///             format!("PIX transfer of {} initiated", params.amount),
///         ))
///     }
///
///     fn check_status(&self, ctx: &ToolContext<MyCtx>, operation_id: &str)
///         -> impl Stream<Item = ToolStatus<PixTransferStage>> + Send
///     {
///         async_stream::stream! {
///             loop {
///                 let status = ctx.app.pix_service.get_status(operation_id).await;
///                 match status {
///                     PixStatus::Success { id } => {
///                         yield ToolStatus::Completed(ToolResult::success(id));
///                         break;
///                     }
///                     _ => yield ToolStatus::Progress { ... };
///                 }
///                 tokio::time::sleep(Duration::from_millis(500)).await;
///             }
///         }
///     }
/// }
/// ```
pub trait AsyncTool<Ctx>: Send + Sync {
    /// The type of name for this tool.
    type Name: ToolName;
    /// The type of progress stages for this tool.
    type Stage: ProgressStage;

    /// Returns the tool's strongly-typed name.
    fn name(&self) -> Self::Name;

    /// Human-readable display name for UI.
    fn display_name(&self) -> &'static str;

    /// Human-readable description of what the tool does.
    fn description(&self) -> &'static str;

    /// JSON schema for the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Permission tier for this tool.
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    /// Execute the tool. Returns immediately with one of:
    /// - Success/Failed: Operation completed synchronously
    /// - `InProgress`: Operation started, use `check_status()` to stream updates
    ///
    /// # Errors
    /// Returns an error if tool execution fails.
    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Value,
    ) -> impl Future<Output = Result<ToolOutcome>> + Send;

    /// Stream status updates for an in-progress operation.
    /// Must yield until Completed or Failed.
    fn check_status(
        &self,
        ctx: &ToolContext<Ctx>,
        operation_id: &str,
    ) -> impl Stream<Item = ToolStatus<Self::Stage>> + Send;
}

// ============================================================================
// Type-Erased Tool (for Registry)
// ============================================================================

/// Type-erased tool trait for registry storage.
///
/// This allows tools with different `Name` associated types to be stored
/// in the same registry by erasing the type information.
///
/// # Example
///
/// ```ignore
/// for tool in registry.all() {
///     println!("Tool: {} - {}", tool.name_str(), tool.description());
/// }
/// ```
#[async_trait]
pub trait ErasedTool<Ctx>: Send + Sync {
    /// Get the tool name as a string.
    fn name_str(&self) -> &str;
    /// Get a human-friendly display name for the tool.
    fn display_name(&self) -> &'static str;
    /// Get the tool description.
    fn description(&self) -> &'static str;
    /// Get the JSON schema for tool inputs.
    fn input_schema(&self) -> Value;
    /// Get the tool's permission tier.
    fn tier(&self) -> ToolTier;
    /// Execute the tool with the given input.
    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult>;
}

/// Wrapper that erases the Name associated type from a Tool.
struct ToolWrapper<T, Ctx>
where
    T: Tool<Ctx>,
{
    inner: T,
    name_cache: String,
    _marker: PhantomData<Ctx>,
}

impl<T, Ctx> ToolWrapper<T, Ctx>
where
    T: Tool<Ctx>,
{
    fn new(tool: T) -> Self {
        let name_cache = tool_name_to_string(&tool.name());
        Self {
            inner: tool,
            name_cache,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<T, Ctx> ErasedTool<Ctx> for ToolWrapper<T, Ctx>
where
    T: Tool<Ctx> + 'static,
    Ctx: Send + Sync + 'static,
{
    fn name_str(&self) -> &str {
        &self.name_cache
    }

    fn display_name(&self) -> &'static str {
        self.inner.display_name()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    fn tier(&self) -> ToolTier {
        self.inner.tier()
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        self.inner.execute(ctx, input).await
    }
}

// ============================================================================
// Type-Erased AsyncTool (for Registry)
// ============================================================================

/// Type-erased async tool trait for registry storage.
///
/// This allows async tools with different `Name` and `Stage` associated types
/// to be stored in the same registry by erasing the type information.
#[async_trait]
pub trait ErasedAsyncTool<Ctx>: Send + Sync {
    /// Get the tool name as a string.
    fn name_str(&self) -> &str;
    /// Get a human-friendly display name for the tool.
    fn display_name(&self) -> &'static str;
    /// Get the tool description.
    fn description(&self) -> &'static str;
    /// Get the JSON schema for tool inputs.
    fn input_schema(&self) -> Value;
    /// Get the tool's permission tier.
    fn tier(&self) -> ToolTier;
    /// Execute the tool with the given input.
    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolOutcome>;
    /// Stream status updates for an in-progress operation (type-erased).
    fn check_status_stream<'a>(
        &'a self,
        ctx: &'a ToolContext<Ctx>,
        operation_id: &'a str,
    ) -> Pin<Box<dyn Stream<Item = ErasedToolStatus> + Send + 'a>>;
}

/// Wrapper that erases the Name and Stage associated types from an [`AsyncTool`].
struct AsyncToolWrapper<T, Ctx>
where
    T: AsyncTool<Ctx>,
{
    inner: T,
    name_cache: String,
    _marker: PhantomData<Ctx>,
}

impl<T, Ctx> AsyncToolWrapper<T, Ctx>
where
    T: AsyncTool<Ctx>,
{
    fn new(tool: T) -> Self {
        let name_cache = tool_name_to_string(&tool.name());
        Self {
            inner: tool,
            name_cache,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<T, Ctx> ErasedAsyncTool<Ctx> for AsyncToolWrapper<T, Ctx>
where
    T: AsyncTool<Ctx> + 'static,
    Ctx: Send + Sync + 'static,
{
    fn name_str(&self) -> &str {
        &self.name_cache
    }

    fn display_name(&self) -> &'static str {
        self.inner.display_name()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    fn tier(&self) -> ToolTier {
        self.inner.tier()
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolOutcome> {
        self.inner.execute(ctx, input).await
    }

    fn check_status_stream<'a>(
        &'a self,
        ctx: &'a ToolContext<Ctx>,
        operation_id: &'a str,
    ) -> Pin<Box<dyn Stream<Item = ErasedToolStatus> + Send + 'a>> {
        use futures::StreamExt;
        let stream = self.inner.check_status(ctx, operation_id);
        Box::pin(stream.map(ErasedToolStatus::from))
    }
}

// ============================================================================
// Tool Registry
// ============================================================================

/// Registry of available tools.
///
/// Tools are stored with their names erased to allow different `Name` types
/// in the same registry. The registry uses string-based lookup for LLM
/// compatibility.
///
/// Supports both synchronous [`Tool`]s and asynchronous [`AsyncTool`]s.
pub struct ToolRegistry<Ctx> {
    tools: HashMap<String, Arc<dyn ErasedTool<Ctx>>>,
    async_tools: HashMap<String, Arc<dyn ErasedAsyncTool<Ctx>>>,
}

impl<Ctx> Clone for ToolRegistry<Ctx> {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
            async_tools: self.async_tools.clone(),
        }
    }
}

impl<Ctx: Send + Sync + 'static> Default for ToolRegistry<Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Ctx: Send + Sync + 'static> ToolRegistry<Ctx> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            async_tools: HashMap::new(),
        }
    }

    /// Register a synchronous tool in the registry.
    ///
    /// The tool's name is converted to a string via serde serialization
    /// and used as the lookup key.
    pub fn register<T>(&mut self, tool: T) -> &mut Self
    where
        T: Tool<Ctx> + 'static,
    {
        let wrapper = ToolWrapper::new(tool);
        let name = wrapper.name_str().to_string();
        self.tools.insert(name, Arc::new(wrapper));
        self
    }

    /// Register an async tool in the registry.
    ///
    /// Async tools have two phases: execute (lightweight, starts operation)
    /// and `check_status` (streams progress until completion).
    pub fn register_async<T>(&mut self, tool: T) -> &mut Self
    where
        T: AsyncTool<Ctx> + 'static,
    {
        let wrapper = AsyncToolWrapper::new(tool);
        let name = wrapper.name_str().to_string();
        self.async_tools.insert(name, Arc::new(wrapper));
        self
    }

    /// Get a synchronous tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ErasedTool<Ctx>>> {
        self.tools.get(name)
    }

    /// Get an async tool by name.
    #[must_use]
    pub fn get_async(&self, name: &str) -> Option<&Arc<dyn ErasedAsyncTool<Ctx>>> {
        self.async_tools.get(name)
    }

    /// Check if a tool name refers to an async tool.
    #[must_use]
    pub fn is_async(&self, name: &str) -> bool {
        self.async_tools.contains_key(name)
    }

    /// Get all registered synchronous tools.
    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn ErasedTool<Ctx>>> {
        self.tools.values()
    }

    /// Get all registered async tools.
    pub fn all_async(&self) -> impl Iterator<Item = &Arc<dyn ErasedAsyncTool<Ctx>>> {
        self.async_tools.values()
    }

    /// Get the number of registered tools (sync + async).
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len() + self.async_tools.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty() && self.async_tools.is_empty()
    }

    /// Filter tools by a predicate.
    ///
    /// Removes tools for which the predicate returns false.
    /// The predicate receives the tool name.
    /// Applies to both sync and async tools.
    ///
    /// # Example
    ///
    /// ```ignore
    /// registry.filter(|name| name != "bash");
    /// ```
    pub fn filter<F>(&mut self, predicate: F)
    where
        F: Fn(&str) -> bool,
    {
        self.tools.retain(|name, _| predicate(name));
        self.async_tools.retain(|name, _| predicate(name));
    }

    /// Convert all tools (sync + async) to LLM tool definitions.
    #[must_use]
    pub fn to_llm_tools(&self) -> Vec<llm::Tool> {
        let mut tools: Vec<_> = self
            .tools
            .values()
            .map(|tool| llm::Tool {
                name: tool.name_str().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.input_schema(),
            })
            .collect();

        tools.extend(self.async_tools.values().map(|tool| llm::Tool {
            name: tool.name_str().to_string(),
            description: tool.description().to_string(),
            input_schema: tool.input_schema(),
        }));

        tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test tool name enum for tests
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum TestToolName {
        MockTool,
        AnotherTool,
    }

    impl ToolName for TestToolName {}

    struct MockTool;

    impl Tool<()> for MockTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::MockTool
        }

        fn display_name(&self) -> &'static str {
            "Mock Tool"
        }

        fn description(&self) -> &'static str {
            "A mock tool for testing"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                }
            })
        }

        async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
            let message = input
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("no message");
            Ok(ToolResult::success(format!("Received: {message}")))
        }
    }

    #[test]
    fn test_tool_name_serialization() {
        let name = TestToolName::MockTool;
        assert_eq!(tool_name_to_string(&name), "mock_tool");

        let parsed: TestToolName = tool_name_from_str("mock_tool").unwrap();
        assert_eq!(parsed, TestToolName::MockTool);
    }

    #[test]
    fn test_dynamic_tool_name() {
        let name = DynamicToolName::new("my_mcp_tool");
        assert_eq!(tool_name_to_string(&name), "my_mcp_tool");
        assert_eq!(name.as_str(), "my_mcp_tool");
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);

        assert_eq!(registry.len(), 1);
        assert!(registry.get("mock_tool").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_to_llm_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);

        let llm_tools = registry.to_llm_tools();
        assert_eq!(llm_tools.len(), 1);
        assert_eq!(llm_tools[0].name, "mock_tool");
    }

    struct AnotherTool;

    impl Tool<()> for AnotherTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::AnotherTool
        }

        fn display_name(&self) -> &'static str {
            "Another Tool"
        }

        fn description(&self) -> &'static str {
            "Another tool for testing"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({ "type": "object" })
        }

        async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
            Ok(ToolResult::success("Done"))
        }
    }

    #[test]
    fn test_filter_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);
        registry.register(AnotherTool);

        assert_eq!(registry.len(), 2);

        // Filter out mock_tool
        registry.filter(|name| name != "mock_tool");

        assert_eq!(registry.len(), 1);
        assert!(registry.get("mock_tool").is_none());
        assert!(registry.get("another_tool").is_some());
    }

    #[test]
    fn test_filter_tools_keep_all() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);
        registry.register(AnotherTool);

        registry.filter(|_| true);

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_filter_tools_remove_all() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);
        registry.register(AnotherTool);

        registry.filter(|_| false);

        assert!(registry.is_empty());
    }

    #[test]
    fn test_display_name() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);

        let tool = registry.get("mock_tool").unwrap();
        assert_eq!(tool.display_name(), "Mock Tool");
    }
}
