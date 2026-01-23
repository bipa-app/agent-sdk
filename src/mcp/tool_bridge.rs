//! Bridge MCP tools to SDK Tool trait.

use crate::tools::{DynamicToolName, Tool, ToolContext, ToolRegistry};
use crate::types::{ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde_json::Value;
use std::fmt::Write;
use std::sync::Arc;

use super::client::McpClient;
use super::protocol::{McpContent, McpToolDefinition};
use super::transport::McpTransport;

/// Bridge an MCP tool to the SDK Tool trait.
///
/// This wrapper allows MCP tools to be used as regular SDK tools.
///
/// # Example
///
/// ```ignore
/// use agent_sdk::mcp::{McpClient, McpToolBridge, StdioTransport};
///
/// let transport = StdioTransport::spawn("npx", &["-y", "mcp-server"]).await?;
/// let client = Arc::new(McpClient::new(transport, "server".to_string()).await?);
///
/// let tools = client.list_tools().await?;
/// for tool_def in tools {
///     let tool = McpToolBridge::new(Arc::clone(&client), tool_def);
///     registry.register(tool);
/// }
/// ```
pub struct McpToolBridge<T: McpTransport> {
    client: Arc<McpClient<T>>,
    definition: McpToolDefinition,
    tier: ToolTier,
}

impl<T: McpTransport> McpToolBridge<T> {
    /// Create a new MCP tool bridge.
    #[must_use]
    pub const fn new(client: Arc<McpClient<T>>, definition: McpToolDefinition) -> Self {
        Self {
            client,
            definition,
            tier: ToolTier::Confirm, // Default to Confirm for safety
        }
    }

    /// Set the tool tier.
    #[must_use]
    pub const fn with_tier(mut self, tier: ToolTier) -> Self {
        self.tier = tier;
        self
    }

    /// Get the tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.definition.name
    }

    /// Get the tool definition.
    #[must_use]
    pub const fn definition(&self) -> &McpToolDefinition {
        &self.definition
    }
}

impl<T: McpTransport + 'static> Tool<()> for McpToolBridge<T> {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new(&self.definition.name)
    }

    fn display_name(&self) -> &'static str {
        // We need to leak the string to get a 'static lifetime
        // This is acceptable since tool definitions are typically long-lived
        Box::leak(self.definition.name.clone().into_boxed_str())
    }

    fn description(&self) -> &'static str {
        let desc = self.definition.description.clone().unwrap_or_default();
        Box::leak(desc.into_boxed_str())
    }

    fn input_schema(&self) -> Value {
        self.definition.input_schema.clone()
    }

    fn tier(&self) -> ToolTier {
        self.tier
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let result = self.client.call_tool(&self.definition.name, input).await?;

        // Convert MCP content to output string
        let output = format_mcp_content(&result.content);

        Ok(ToolResult {
            success: !result.is_error,
            output,
            data: Some(serde_json::to_value(&result).unwrap_or_default()),
            duration_ms: None,
        })
    }
}

/// Format MCP content items as a string.
fn format_mcp_content(content: &[McpContent]) -> String {
    let mut output = String::new();

    for item in content {
        match item {
            McpContent::Text { text } => {
                output.push_str(text);
                output.push('\n');
            }
            McpContent::Image { mime_type, .. } => {
                let _ = writeln!(output, "[Image: {mime_type}]");
            }
            McpContent::Resource { uri, text, .. } => {
                if let Some(text) = text {
                    output.push_str(text);
                    output.push('\n');
                } else {
                    let _ = writeln!(output, "[Resource: {uri}]");
                }
            }
        }
    }

    output.trim_end().to_string()
}

/// Register all tools from an MCP client into a tool registry.
///
/// # Arguments
///
/// * `registry` - The tool registry to add tools to
/// * `client` - The MCP client to get tools from
///
/// # Errors
///
/// Returns an error if listing tools fails.
///
/// # Example
///
/// ```ignore
/// use agent_sdk::mcp::{register_mcp_tools, McpClient, StdioTransport};
/// use agent_sdk::ToolRegistry;
///
/// let transport = StdioTransport::spawn("npx", &["-y", "mcp-server"]).await?;
/// let client = Arc::new(McpClient::new(transport, "server".to_string()).await?);
///
/// let mut registry = ToolRegistry::new();
/// register_mcp_tools(&mut registry, client).await?;
/// ```
pub async fn register_mcp_tools<T: McpTransport + 'static>(
    registry: &mut ToolRegistry<()>,
    client: Arc<McpClient<T>>,
) -> Result<()> {
    let tools = client
        .list_tools()
        .await
        .context("Failed to list MCP tools")?;

    for definition in tools {
        let bridge = McpToolBridge::new(Arc::clone(&client), definition);
        registry.register(bridge);
    }

    Ok(())
}

/// Register MCP tools with custom tier assignment.
///
/// # Arguments
///
/// * `registry` - The tool registry to add tools to
/// * `client` - The MCP client to get tools from
/// * `tier_fn` - Function to determine tier for each tool
///
/// # Errors
///
/// Returns an error if listing tools fails.
pub async fn register_mcp_tools_with_tiers<T, F>(
    registry: &mut ToolRegistry<()>,
    client: Arc<McpClient<T>>,
    tier_fn: F,
) -> Result<()>
where
    T: McpTransport + 'static,
    F: Fn(&McpToolDefinition) -> ToolTier,
{
    let tools = client
        .list_tools()
        .await
        .context("Failed to list MCP tools")?;

    for definition in tools {
        let tier = tier_fn(&definition);
        let bridge = McpToolBridge::new(Arc::clone(&client), definition).with_tier(tier);
        registry.register(bridge);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_mcp_content_text() {
        let content = vec![McpContent::Text {
            text: "Hello, world!".to_string(),
        }];

        let output = format_mcp_content(&content);
        assert_eq!(output, "Hello, world!");
    }

    #[test]
    fn test_format_mcp_content_multiple() {
        let content = vec![
            McpContent::Text {
                text: "First line".to_string(),
            },
            McpContent::Text {
                text: "Second line".to_string(),
            },
        ];

        let output = format_mcp_content(&content);
        assert_eq!(output, "First line\nSecond line");
    }

    #[test]
    fn test_format_mcp_content_image() {
        let content = vec![McpContent::Image {
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
        }];

        let output = format_mcp_content(&content);
        assert_eq!(output, "[Image: image/png]");
    }

    #[test]
    fn test_format_mcp_content_resource() {
        let content = vec![McpContent::Resource {
            uri: "file:///path/to/file".to_string(),
            mime_type: Some("text/plain".to_string()),
            text: None,
        }];

        let output = format_mcp_content(&content);
        assert!(output.contains("file:///path/to/file"));
    }

    #[test]
    fn test_format_mcp_content_resource_with_text() {
        let content = vec![McpContent::Resource {
            uri: "file:///path/to/file".to_string(),
            mime_type: Some("text/plain".to_string()),
            text: Some("File contents".to_string()),
        }];

        let output = format_mcp_content(&content);
        assert_eq!(output, "File contents");
    }

    #[test]
    fn test_format_mcp_content_empty() {
        let content: Vec<McpContent> = vec![];
        let output = format_mcp_content(&content);
        assert!(output.is_empty());
    }
}
