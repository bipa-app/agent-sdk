//! MCP client implementation.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::sync::Arc;

use super::protocol::JsonRpcRequest;
use super::protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, InitializeResult, McpToolCallResult,
    McpToolDefinition, ToolCallParams, ToolsListResult,
};
use super::transport::McpTransport;

/// MCP protocol version.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// MCP client for communicating with MCP servers.
///
/// The client handles the MCP protocol, including initialization,
/// tool discovery, and tool execution.
///
/// # Example
///
/// ```ignore
/// use agent_sdk::mcp::{McpClient, StdioTransport};
///
/// // Spawn server and create client
/// let transport = StdioTransport::spawn("npx", &["-y", "mcp-server"]).await?;
/// let client = McpClient::new(transport, "my-server".to_string()).await?;
///
/// // List available tools
/// let tools = client.list_tools().await?;
///
/// // Call a tool
/// let result = client.call_tool("tool_name", json!({"arg": "value"})).await?;
/// ```
pub struct McpClient<T: McpTransport> {
    transport: Arc<T>,
    server_name: String,
    server_info: Option<InitializeResult>,
}

impl<T: McpTransport> McpClient<T> {
    /// Create a new MCP client and initialize the connection.
    ///
    /// # Arguments
    ///
    /// * `transport` - The transport to use for communication
    /// * `server_name` - A name to identify this server connection
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub async fn new(transport: Arc<T>, server_name: String) -> Result<Self> {
        let mut client = Self {
            transport,
            server_name,
            server_info: None,
        };

        client.initialize().await?;

        Ok(client)
    }

    /// Create a client without initialization.
    ///
    /// Use this if you need to control when initialization happens.
    #[must_use]
    pub const fn new_uninitialized(transport: Arc<T>, server_name: String) -> Self {
        Self {
            transport,
            server_name,
            server_info: None,
        }
    }

    /// Initialize the MCP connection.
    ///
    /// This must be called before using other methods.
    ///
    /// # Errors
    ///
    /// Returns an error if the server rejects initialization.
    pub async fn initialize(&mut self) -> Result<&InitializeResult> {
        #[cfg(feature = "otel")]
        let mut span = start_mcp_span("mcp.initialize", &self.server_name);

        let result = self.initialize_inner().await;

        #[cfg(feature = "otel")]
        finish_mcp_span(&mut span, &result);

        result?;

        self.server_info
            .as_ref()
            .context("Server info not available")
    }

    async fn initialize_inner(&mut self) -> Result<()> {
        let params = InitializeParams {
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "agent-sdk".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        let request = JsonRpcRequest::new("initialize", Some(serde_json::to_value(&params)?), 0);

        let response = self.transport.send(request).await?;

        let result: InitializeResult = response
            .result
            .map(serde_json::from_value)
            .transpose()
            .context("Failed to parse initialize response")?
            .context("Initialize response missing result")?;

        // Send initialized notification (fire-and-forget)
        let notification = JsonRpcRequest::new("notifications/initialized", None, 0);
        let _ = self.transport.send_notification(notification).await;

        self.server_info = Some(result);
        Ok(())
    }

    /// Get the server name.
    #[must_use]
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Get server info if initialized.
    #[must_use]
    pub const fn server_info(&self) -> Option<&InitializeResult> {
        self.server_info.as_ref()
    }

    /// List available tools from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>> {
        #[cfg(feature = "otel")]
        let mut span = start_mcp_span("mcp.tools/list", &self.server_name);

        let result = self.list_tools_inner().await;

        #[cfg(feature = "otel")]
        {
            use opentelemetry::KeyValue;
            use opentelemetry::trace::Span;
            if let Ok(ref tools) = result {
                span.set_attribute(KeyValue::new(
                    "mcp.tools.count",
                    i64::try_from(tools.len()).unwrap_or(0),
                ));
            }
            finish_mcp_span(&mut span, &result);
        }

        result
    }

    async fn list_tools_inner(&self) -> Result<Vec<McpToolDefinition>> {
        let request = JsonRpcRequest::new("tools/list", None, 0);

        let response = self.transport.send(request).await?;

        let result: ToolsListResult = response
            .result
            .map(serde_json::from_value)
            .transpose()
            .context("Failed to parse tools/list response")?
            .context("tools/list response missing result")?;

        Ok(result.tools)
    }

    /// Call a tool on the server.
    ///
    /// # Arguments
    ///
    /// * `name` - Tool name to call
    /// * `arguments` - Tool arguments as JSON
    ///
    /// # Errors
    ///
    /// Returns an error if the tool call fails.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<McpToolCallResult> {
        #[cfg(feature = "otel")]
        let mut span = {
            use opentelemetry::KeyValue;
            start_mcp_span_with_attrs(
                "mcp.tools/call",
                vec![
                    KeyValue::new("mcp.server.name", self.server_name.clone()),
                    KeyValue::new("gen_ai.tool.name", name.to_string()),
                ],
            )
        };

        let result = self.call_tool_inner(name, arguments).await;

        #[cfg(feature = "otel")]
        finish_mcp_call_tool_span(&mut span, &result);

        result
    }

    async fn call_tool_inner(&self, name: &str, arguments: Value) -> Result<McpToolCallResult> {
        let params = ToolCallParams {
            name: name.to_string(),
            arguments: Some(arguments),
        };

        let request = JsonRpcRequest::new("tools/call", Some(serde_json::to_value(&params)?), 0);

        let response = self.transport.send(request).await?;

        if let Some(ref error) = response.error {
            bail!("Tool call failed: {} (code {})", error.message, error.code);
        }

        let result: McpToolCallResult = response
            .result
            .map(serde_json::from_value)
            .transpose()
            .context("Failed to parse tools/call response")?
            .context("tools/call response missing result")?;

        Ok(result)
    }

    /// Call a tool with raw Value arguments.
    ///
    /// # Arguments
    ///
    /// * `name` - Tool name to call
    /// * `arguments` - Tool arguments as optional JSON
    ///
    /// # Errors
    ///
    /// Returns an error if the tool call fails.
    pub async fn call_tool_raw(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<McpToolCallResult> {
        let args = arguments.unwrap_or_else(|| json!({}));
        self.call_tool(name, args).await
    }

    /// Close the client connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails to close.
    pub async fn close(&self) -> Result<()> {
        self.transport.close().await
    }
}

#[cfg(feature = "otel")]
fn start_mcp_span(
    name: impl Into<std::borrow::Cow<'static, str>>,
    server_name: &str,
) -> opentelemetry::global::BoxedSpan {
    use opentelemetry::KeyValue;
    start_mcp_span_with_attrs(
        name,
        vec![KeyValue::new("mcp.server.name", server_name.to_string())],
    )
}

#[cfg(feature = "otel")]
fn start_mcp_span_with_attrs(
    name: impl Into<std::borrow::Cow<'static, str>>,
    attrs: Vec<opentelemetry::KeyValue>,
) -> opentelemetry::global::BoxedSpan {
    use crate::observability::spans;
    spans::start_client_span(name, attrs)
}

#[cfg(feature = "otel")]
fn finish_mcp_span<T>(span: &mut opentelemetry::global::BoxedSpan, result: &Result<T>) {
    use crate::observability::spans;
    use opentelemetry::trace::Span;

    if let Err(err) = result {
        spans::set_span_error(span, "mcp_error", &format!("{err}"));
    }
    span.end();
}

#[cfg(feature = "otel")]
fn finish_mcp_call_tool_span(
    span: &mut opentelemetry::global::BoxedSpan,
    result: &Result<super::protocol::McpToolCallResult>,
) {
    use crate::observability::spans;
    use opentelemetry::trace::Span;

    match result {
        Ok(tool_result) if tool_result.is_error => {
            let error_text = tool_result
                .content
                .iter()
                .find_map(|c| match c {
                    super::protocol::McpContent::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or("MCP tool returned error");
            spans::set_span_error(span, "tool_error", error_text);
        }
        Err(err) => {
            spans::set_span_error(span, "mcp_error", &format!("{err}"));
        }
        Ok(_) => {}
    }
    span.end();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_protocol_version() {
        assert!(!MCP_PROTOCOL_VERSION.is_empty());
    }

    #[test]
    fn test_client_info() {
        let info = ClientInfo {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
        };

        let json = serde_json::to_string(&info).expect("serialize");
        assert!(json.contains("test"));
        assert!(json.contains("1.0.0"));
    }
}
