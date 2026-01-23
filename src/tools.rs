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

use crate::events::AgentEvent;
use crate::llm;
use crate::types::{ToolResult, ToolTier};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
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

/// Context passed to tool execution
pub struct ToolContext<Ctx> {
    /// Application-specific context (e.g., `user_id`, db connection)
    pub app: Ctx,
    /// Tool-specific metadata
    pub metadata: HashMap<String, Value>,
    /// Optional channel for tools to emit events (e.g., subagent progress)
    event_tx: Option<mpsc::Sender<AgentEvent>>,
}

impl<Ctx> ToolContext<Ctx> {
    #[must_use]
    pub fn new(app: Ctx) -> Self {
        Self {
            app,
            metadata: HashMap::new(),
            event_tx: None,
        }
    }

    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set the event channel for tools that need to emit events during execution.
    #[must_use]
    pub fn with_event_tx(mut self, tx: mpsc::Sender<AgentEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// Emit an event through the event channel (if set).
    ///
    /// This uses `try_send` to avoid blocking and to ensure the future is `Send`.
    /// The event is silently dropped if the channel is full.
    pub fn emit_event(&self, event: AgentEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.try_send(event);
        }
    }

    /// Get a clone of the event channel sender (if set).
    ///
    /// This is useful for tools that spawn subprocesses (like subagents)
    /// and need to forward events to the parent's event stream.
    #[must_use]
    pub fn event_tx(&self) -> Option<mpsc::Sender<AgentEvent>> {
        self.event_tx.clone()
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
// Tool Registry
// ============================================================================

/// Registry of available tools.
///
/// Tools are stored with their names erased to allow different `Name` types
/// in the same registry. The registry uses string-based lookup for LLM
/// compatibility.
pub struct ToolRegistry<Ctx> {
    tools: HashMap<String, Arc<dyn ErasedTool<Ctx>>>,
}

impl<Ctx> Clone for ToolRegistry<Ctx> {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
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
        }
    }

    /// Register a tool in the registry.
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

    /// Get a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ErasedTool<Ctx>>> {
        self.tools.get(name)
    }

    /// Get all registered tools.
    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn ErasedTool<Ctx>>> {
        self.tools.values()
    }

    /// Get the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Filter tools by a predicate.
    ///
    /// Removes tools for which the predicate returns false.
    /// The predicate receives the tool name.
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
    }

    /// Convert tools to LLM tool definitions.
    #[must_use]
    pub fn to_llm_tools(&self) -> Vec<llm::Tool> {
        self.tools
            .values()
            .map(|tool| llm::Tool {
                name: tool.name_str().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.input_schema(),
            })
            .collect()
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
