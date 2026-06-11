//! MCP client implementation.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::sync::Arc;

use super::protocol::JsonRpcRequest;
use super::protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, InitializeResult, McpPrompt, McpResource,
    McpToolCallResult, McpToolDefinition, PREFERRED_PROTOCOL_VERSION, PromptGetParams,
    PromptGetResult, PromptsListResult, ResourceReadParams, ResourceReadResult,
    ResourcesListResult, ToolCallParams, ToolsListResult, is_known_protocol_version,
};
use super::transport::McpTransport;

/// MCP protocol revision this client advertises during `initialize`.
///
/// Retained as a public alias of [`PREFERRED_PROTOCOL_VERSION`] for backwards
/// compatibility. The revision actually used for a connection is whatever the
/// server selects during the handshake — see [`McpClient::protocol_version`].
pub const MCP_PROTOCOL_VERSION: &str = PREFERRED_PROTOCOL_VERSION;

/// Upper bound on pages followed by a cursor-paginated list call.
///
/// Bounds the loop in case a misbehaving server returns a non-empty cursor
/// forever; legitimate servers terminate well before this.
const MAX_LIST_PAGES: usize = 10_000;

/// Decide whether to continue paginating.
///
/// Returns `Ok(Some(cursor))` to fetch another page, `Ok(None)` to stop (no
/// further cursor), and increments/guards the page counter. Errors if the
/// server exceeds [`MAX_LIST_PAGES`] (likely a cursor that never terminates).
fn next_cursor_or_stop(next: Option<String>, pages: &mut usize) -> Result<Option<String>> {
    *pages += 1;
    match next {
        Some(cursor) if !cursor.is_empty() => {
            if *pages >= MAX_LIST_PAGES {
                bail!("MCP list pagination exceeded {MAX_LIST_PAGES} pages; aborting");
            }
            Ok(Some(cursor))
        }
        _ => Ok(None),
    }
}

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
    /// Protocol revision selected by the server during `initialize`.
    negotiated_version: Option<String>,
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
            negotiated_version: None,
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
            negotiated_version: None,
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
        let started_at = std::time::Instant::now();
        #[cfg(feature = "otel")]
        let mut span = {
            use crate::observability::langfuse;
            let mut span = start_mcp_span("mcp.initialize", &self.server_name);
            langfuse::tag_observation(&mut span, langfuse::ObservationType::Chain);
            span
        };

        let result = self.initialize_inner().await;

        #[cfg(feature = "otel")]
        finish_mcp_span(
            &mut span,
            &result,
            "initialize",
            &self.server_name,
            started_at,
        );

        result?;

        self.server_info
            .as_ref()
            .context("Server info not available")
    }

    async fn initialize_inner(&mut self) -> Result<()> {
        let params = InitializeParams {
            protocol_version: PREFERRED_PROTOCOL_VERSION.to_string(),
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

        // Honour the revision the server actually selected. The server may
        // downgrade to an older revision (e.g. a legacy `2024-11-05` server);
        // we adapt to its choice rather than insisting on our preference. An
        // unrecognised revision is not fatal — proceed but log it.
        let negotiated = result.protocol_version.clone();
        if !is_known_protocol_version(&negotiated) {
            log::warn!(
                "MCP server '{}' negotiated unknown protocol revision '{}' (advertised '{}')",
                self.server_name,
                negotiated,
                PREFERRED_PROTOCOL_VERSION,
            );
        }
        // Inform the transport so out-of-band carriers (HTTP header) can use it.
        self.transport.set_protocol_version(&negotiated).await;
        self.negotiated_version = Some(negotiated);

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

    /// The MCP protocol revision negotiated with the server.
    ///
    /// Returns `None` until [`McpClient::initialize`] has completed. This is
    /// the revision the *server* selected, which may be older than
    /// [`PREFERRED_PROTOCOL_VERSION`] if the server is on a legacy build.
    #[must_use]
    pub fn protocol_version(&self) -> Option<&str> {
        self.negotiated_version.as_deref()
    }

    /// List available tools from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>> {
        #[cfg(feature = "otel")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "otel")]
        let mut span = {
            use crate::observability::langfuse;
            let mut span = start_mcp_span("mcp.tools/list", &self.server_name);
            langfuse::tag_observation(&mut span, langfuse::ObservationType::Chain);
            span
        };

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
            finish_mcp_span(
                &mut span,
                &result,
                "tools/list",
                &self.server_name,
                started_at,
            );
        }

        result
    }

    async fn list_tools_inner(&self) -> Result<Vec<McpToolDefinition>> {
        let mut all = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0usize;

        loop {
            let params = cursor.as_ref().map(|c| json!({ "cursor": c }));
            let request = JsonRpcRequest::new("tools/list", params, 0);
            let response = self.transport.send(request).await?;

            let value = response
                .result
                .context("tools/list response missing result")?;
            // `ToolsListResult` does not model `nextCursor`, so read it from the
            // raw value before deserializing the typed page.
            let next = value
                .get("nextCursor")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string);
            let result: ToolsListResult =
                serde_json::from_value(value).context("Failed to parse tools/list response")?;
            all.extend(result.tools);

            match next_cursor_or_stop(next, &mut pages)? {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }

        Ok(all)
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
        let started_at = std::time::Instant::now();
        #[cfg(feature = "otel")]
        let mut span = {
            use crate::observability::langfuse;
            use opentelemetry::KeyValue;
            let mut span = start_mcp_span_with_attrs(
                "mcp.tools/call",
                vec![
                    KeyValue::new("mcp.server.name", self.server_name.clone()),
                    KeyValue::new("gen_ai.tool.name", name.to_string()),
                ],
            );
            langfuse::tag_observation(&mut span, langfuse::ObservationType::Tool);
            span
        };

        let result = self.call_tool_inner(name, arguments).await;

        #[cfg(feature = "otel")]
        finish_mcp_call_tool_span(
            &mut span,
            &result,
            "tools/call",
            &self.server_name,
            started_at,
        );

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

    /// List resources exposed by the server (`resources/list`).
    ///
    /// Resources are addressable data (files, database rows, API payloads) the
    /// server makes available for reading. Returns an empty list if the server
    /// did not advertise the `resources` capability.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the response cannot be parsed.
    pub async fn list_resources(&self) -> Result<Vec<McpResource>> {
        if !self.supports_resources() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "otel")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "otel")]
        let mut span = {
            use crate::observability::langfuse;
            let mut span = start_mcp_span("mcp.resources/list", &self.server_name);
            langfuse::tag_observation(&mut span, langfuse::ObservationType::Chain);
            span
        };

        let result = self.list_resources_inner().await;

        #[cfg(feature = "otel")]
        finish_mcp_span(
            &mut span,
            &result,
            "resources/list",
            &self.server_name,
            started_at,
        );

        result
    }

    async fn list_resources_inner(&self) -> Result<Vec<McpResource>> {
        let mut all = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0usize;

        loop {
            let params = cursor.as_ref().map(|c| json!({ "cursor": c }));
            let request = JsonRpcRequest::new("resources/list", params, 0);
            let response = self.transport.send(request).await?;
            let result: ResourcesListResult = response
                .result
                .map(serde_json::from_value)
                .transpose()
                .context("Failed to parse resources/list response")?
                .context("resources/list response missing result")?;
            all.extend(result.resources);

            match next_cursor_or_stop(result.next_cursor, &mut pages)? {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }

        Ok(all)
    }

    /// Read a resource by URI (`resources/read`).
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the response cannot be parsed.
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceReadResult> {
        #[cfg(feature = "otel")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "otel")]
        let mut span = {
            use crate::observability::langfuse;
            let mut span = start_mcp_span("mcp.resources/read", &self.server_name);
            langfuse::tag_observation(&mut span, langfuse::ObservationType::Chain);
            span
        };

        let result = self.read_resource_inner(uri).await;

        #[cfg(feature = "otel")]
        finish_mcp_span(
            &mut span,
            &result,
            "resources/read",
            &self.server_name,
            started_at,
        );

        result
    }

    async fn read_resource_inner(&self, uri: &str) -> Result<ResourceReadResult> {
        let params = ResourceReadParams {
            uri: uri.to_string(),
        };
        let request =
            JsonRpcRequest::new("resources/read", Some(serde_json::to_value(&params)?), 0);
        let response = self.transport.send(request).await?;
        let result: ResourceReadResult = response
            .result
            .map(serde_json::from_value)
            .transpose()
            .context("Failed to parse resources/read response")?
            .context("resources/read response missing result")?;
        Ok(result)
    }

    /// List prompts exposed by the server (`prompts/list`).
    ///
    /// Returns an empty list if the server did not advertise the `prompts`
    /// capability.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the response cannot be parsed.
    pub async fn list_prompts(&self) -> Result<Vec<McpPrompt>> {
        if !self.supports_prompts() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "otel")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "otel")]
        let mut span = {
            use crate::observability::langfuse;
            let mut span = start_mcp_span("mcp.prompts/list", &self.server_name);
            langfuse::tag_observation(&mut span, langfuse::ObservationType::Chain);
            span
        };

        let result = self.list_prompts_inner().await;

        #[cfg(feature = "otel")]
        finish_mcp_span(
            &mut span,
            &result,
            "prompts/list",
            &self.server_name,
            started_at,
        );

        result
    }

    async fn list_prompts_inner(&self) -> Result<Vec<McpPrompt>> {
        let mut all = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0usize;

        loop {
            let params = cursor.as_ref().map(|c| json!({ "cursor": c }));
            let request = JsonRpcRequest::new("prompts/list", params, 0);
            let response = self.transport.send(request).await?;
            let result: PromptsListResult = response
                .result
                .map(serde_json::from_value)
                .transpose()
                .context("Failed to parse prompts/list response")?
                .context("prompts/list response missing result")?;
            all.extend(result.prompts);

            match next_cursor_or_stop(result.next_cursor, &mut pages)? {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }

        Ok(all)
    }

    /// Fetch and render a prompt by name (`prompts/get`).
    ///
    /// # Arguments
    ///
    /// * `name` - Prompt name to fetch.
    /// * `arguments` - Optional arguments to interpolate into the template.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the response cannot be parsed.
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<PromptGetResult> {
        #[cfg(feature = "otel")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "otel")]
        let mut span = {
            use crate::observability::langfuse;
            let mut span = start_mcp_span("mcp.prompts/get", &self.server_name);
            langfuse::tag_observation(&mut span, langfuse::ObservationType::Chain);
            span
        };

        let result = self.get_prompt_inner(name, arguments).await;

        #[cfg(feature = "otel")]
        finish_mcp_span(
            &mut span,
            &result,
            "prompts/get",
            &self.server_name,
            started_at,
        );

        result
    }

    async fn get_prompt_inner(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<PromptGetResult> {
        let params = PromptGetParams {
            name: name.to_string(),
            arguments,
        };
        let request = JsonRpcRequest::new("prompts/get", Some(serde_json::to_value(&params)?), 0);
        let response = self.transport.send(request).await?;
        let result: PromptGetResult = response
            .result
            .map(serde_json::from_value)
            .transpose()
            .context("Failed to parse prompts/get response")?
            .context("prompts/get response missing result")?;
        Ok(result)
    }

    /// Whether the server advertised the `resources` capability.
    #[must_use]
    pub fn supports_resources(&self) -> bool {
        self.server_info
            .as_ref()
            .is_some_and(|info| info.capabilities.resources.is_some())
    }

    /// Whether the server advertised the `prompts` capability.
    #[must_use]
    pub fn supports_prompts(&self) -> bool {
        self.server_info
            .as_ref()
            .is_some_and(|info| info.capabilities.prompts.is_some())
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
    use crate::observability::{baggage, spans};
    let mut span = spans::start_client_span(name, attrs);
    baggage::copy_baggage_to_active_span(&mut span);
    span
}

#[cfg(feature = "otel")]
fn finish_mcp_span<T>(
    span: &mut opentelemetry::global::BoxedSpan,
    result: &Result<T>,
    method: &'static str,
    server_name: &str,
    started_at: std::time::Instant,
) {
    use crate::observability::{metrics, spans};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    let mut metric_attrs = vec![
        KeyValue::new("mcp.method", method),
        KeyValue::new("mcp.server.name", server_name.to_string()),
    ];
    if let Err(err) = result {
        spans::set_span_error(span, "mcp_error", &format!("{err}"));
        metric_attrs.push(KeyValue::new(
            crate::observability::attrs::ERROR_TYPE,
            "mcp_error",
        ));
    }
    let elapsed_secs = started_at.elapsed().as_secs_f64();
    metrics::Metrics::global()
        .mcp_requests_duration
        .record(elapsed_secs, &metric_attrs);
    span.end();
}

#[cfg(feature = "otel")]
fn finish_mcp_call_tool_span(
    span: &mut opentelemetry::global::BoxedSpan,
    result: &Result<super::protocol::McpToolCallResult>,
    method: &'static str,
    server_name: &str,
    started_at: std::time::Instant,
) {
    use crate::observability::{metrics, spans};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::Span;

    let mut metric_attrs = vec![
        KeyValue::new("mcp.method", method),
        KeyValue::new("mcp.server.name", server_name.to_string()),
    ];
    let error_kind: Option<&'static str> = match result {
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
            Some("tool_error")
        }
        Err(err) => {
            spans::set_span_error(span, "mcp_error", &format!("{err}"));
            Some("mcp_error")
        }
        Ok(_) => None,
    };
    if let Some(kind) = error_kind {
        metric_attrs.push(KeyValue::new(crate::observability::attrs::ERROR_TYPE, kind));
    }
    let elapsed_secs = started_at.elapsed().as_secs_f64();
    metrics::Metrics::global()
        .mcp_requests_duration
        .record(elapsed_secs, &metric_attrs);
    span.end();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::protocol::JsonRpcResponse;
    use async_trait::async_trait;

    /// Transport that serves a two-page `tools/list`, gated by the `cursor`
    /// param, so pagination can be exercised with no live server.
    struct PagingTransport;

    #[async_trait]
    impl McpTransport for PagingTransport {
        async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
            let cursor = request
                .params
                .as_ref()
                .and_then(|p| p.get("cursor"))
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string);

            let result = match request.method.as_str() {
                "tools/list" => match cursor.as_deref() {
                    None => json!({
                        "tools": [{"name": "alpha", "inputSchema": {"type": "object"}}],
                        "nextCursor": "page2",
                    }),
                    Some("page2") => json!({
                        "tools": [{"name": "beta", "inputSchema": {"type": "object"}}],
                    }),
                    Some(other) => bail!("unexpected cursor {other}"),
                },
                other => bail!("unexpected method {other}"),
            };

            Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(result),
                error: None,
                id: request.id,
            })
        }

        async fn send_notification(&self, _request: JsonRpcRequest) -> Result<()> {
            Ok(())
        }

        async fn close(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Regression test for finding 6: `list_tools` must follow `nextCursor` and
    /// merge every page, not silently return only page 1.
    #[tokio::test]
    async fn list_tools_follows_pagination() -> Result<()> {
        let client = McpClient::new_uninitialized(Arc::new(PagingTransport), "test".to_string());
        let tools = client.list_tools().await?;
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["alpha", "beta"],
            "both pages must be merged in order"
        );
        Ok(())
    }

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
