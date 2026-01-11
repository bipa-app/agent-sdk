use crate::llm;
use crate::types::{ToolResult, ToolTier};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Context passed to tool execution
pub struct ToolContext<Ctx> {
    /// Application-specific context (e.g., `user_id`, db connection)
    pub app: Ctx,
    /// Tool-specific metadata
    pub metadata: HashMap<String, Value>,
}

impl<Ctx> ToolContext<Ctx> {
    #[must_use]
    pub fn new(app: Ctx) -> Self {
        Self {
            app,
            metadata: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Definition of a tool that can be called by the agent
#[async_trait]
pub trait Tool<Ctx>: Send + Sync {
    /// Unique name for the tool (used in LLM tool calls)
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does
    fn description(&self) -> &str;

    /// JSON schema for the tool's input parameters
    fn input_schema(&self) -> Value;

    /// Permission tier for this tool
    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    /// Execute the tool with the given input
    ///
    /// # Errors
    /// Returns an error if tool execution fails.
    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult>;
}

/// Registry of available tools
pub struct ToolRegistry<Ctx> {
    tools: HashMap<String, Arc<dyn Tool<Ctx>>>,
}

impl<Ctx> Default for ToolRegistry<Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Ctx> ToolRegistry<Ctx> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool in the registry
    pub fn register<T: Tool<Ctx> + 'static>(&mut self, tool: T) -> &mut Self {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
        self
    }

    /// Register a boxed tool
    pub fn register_boxed(&mut self, tool: Arc<dyn Tool<Ctx>>) -> &mut Self {
        self.tools.insert(tool.name().to_string(), tool);
        self
    }

    /// Get a tool by name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool<Ctx>>> {
        self.tools.get(name)
    }

    /// Get all registered tools
    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn Tool<Ctx>>> {
        self.tools.values()
    }

    /// Get the number of registered tools
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Convert tools to LLM tool definitions
    #[must_use]
    pub fn to_llm_tools(&self) -> Vec<llm::Tool> {
        self.tools
            .values()
            .map(|tool| llm::Tool {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.input_schema(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool;

    #[async_trait]
    impl Tool<()> for MockTool {
        fn name(&self) -> &'static str {
            "mock_tool"
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
}
