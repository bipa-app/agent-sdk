//! MCP JSON-RPC protocol types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC version string.
pub const JSONRPC_VERSION: &str = "2.0";

/// JSON-RPC request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (always "2.0").
    pub jsonrpc: String,
    /// Request method name.
    pub method: String,
    /// Request parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    /// Request ID.
    pub id: RequestId,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request.
    #[must_use]
    pub fn new(method: impl Into<String>, params: Option<Value>, id: u64) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
            id: RequestId::Number(id),
        }
    }
}

/// JSON-RPC request ID.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum RequestId {
    /// Numeric ID.
    Number(u64),
    /// String ID.
    String(String),
}

/// JSON-RPC response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version (always "2.0").
    pub jsonrpc: String,
    /// Response result (success case).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Response error (error case).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    /// Request ID this response corresponds to.
    pub id: RequestId,
}

impl JsonRpcResponse {
    /// Check if this response is an error.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Get the result value, if present.
    #[must_use]
    pub const fn result(&self) -> Option<&Value> {
        self.result.as_ref()
    }
}

/// JSON-RPC error object.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: i32,
    /// Error message.
    pub message: String,
    /// Additional error data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Standard JSON-RPC error codes.
pub mod error_codes {
    /// Parse error - Invalid JSON.
    pub const PARSE_ERROR: i32 = -32700;
    /// Invalid Request - JSON is not a valid Request object.
    pub const INVALID_REQUEST: i32 = -32600;
    /// Method not found.
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Invalid params.
    pub const INVALID_PARAMS: i32 = -32602;
    /// Internal error.
    pub const INTERNAL_ERROR: i32 = -32603;
}

/// MCP tool definition from server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolDefinition {
    /// Tool name.
    pub name: String,
    /// Tool description.
    #[serde(default)]
    pub description: Option<String>,
    /// Input schema (JSON Schema).
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// MCP tool call result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolCallResult {
    /// Content items returned by the tool.
    pub content: Vec<McpContent>,
    /// Whether this is an error result.
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

/// MCP content item.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
    },
    /// Image content (base64 encoded).
    #[serde(rename = "image")]
    Image {
        /// Base64 encoded image data.
        data: String,
        /// MIME type of the image.
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Resource reference.
    #[serde(rename = "resource")]
    Resource {
        /// Resource URI.
        uri: String,
        /// Resource MIME type.
        #[serde(rename = "mimeType")]
        mime_type: Option<String>,
        /// Optional text content.
        text: Option<String>,
    },
}

/// MCP server capabilities.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    /// Tool capabilities.
    #[serde(default)]
    pub tools: Option<McpToolsCapability>,
    /// Resource capabilities.
    #[serde(default)]
    pub resources: Option<McpResourcesCapability>,
    /// Prompt capabilities.
    #[serde(default)]
    pub prompts: Option<McpPromptsCapability>,
}

/// Tool capabilities.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct McpToolsCapability {
    /// Whether tools list can change.
    #[serde(default, rename = "listChanged")]
    pub list_changed: bool,
}

/// Resource capabilities.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct McpResourcesCapability {
    /// Whether subscriptions are supported.
    #[serde(default)]
    pub subscribe: bool,
    /// Whether resource list can change.
    #[serde(default, rename = "listChanged")]
    pub list_changed: bool,
}

/// Prompt capabilities.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct McpPromptsCapability {
    /// Whether prompts list can change.
    #[serde(default, rename = "listChanged")]
    pub list_changed: bool,
}

/// Initialize request params.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitializeParams {
    /// Protocol version.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Client capabilities.
    pub capabilities: ClientCapabilities,
    /// Client info.
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

/// Client capabilities.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Roots capability.
    #[serde(default)]
    pub roots: Option<RootsCapability>,
    /// Sampling capability.
    #[serde(default)]
    pub sampling: Option<SamplingCapability>,
}

/// Roots capability.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RootsCapability {
    /// Whether list can change.
    #[serde(default, rename = "listChanged")]
    pub list_changed: bool,
}

/// Sampling capability.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SamplingCapability {}

/// Client info.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client name.
    pub name: String,
    /// Client version.
    pub version: String,
}

/// Initialize response result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitializeResult {
    /// Protocol version.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Server capabilities.
    pub capabilities: McpServerCapabilities,
    /// Server info.
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

/// Server info.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name.
    pub name: String,
    /// Server version.
    #[serde(default)]
    pub version: Option<String>,
}

/// Tools list response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolsListResult {
    /// List of available tools.
    pub tools: Vec<McpToolDefinition>,
}

/// Tool call params.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallParams {
    /// Tool name.
    pub name: String,
    /// Tool arguments.
    #[serde(default)]
    pub arguments: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_request_serialization() {
        let request =
            JsonRpcRequest::new("test_method", Some(serde_json::json!({"key": "value"})), 1);

        let json = serde_json::to_string(&request).expect("serialize");
        assert!(json.contains("test_method"));
        assert!(json.contains("2.0"));
    }

    #[test]
    fn test_json_rpc_response_success() {
        let response = JsonRpcResponse {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: Some(serde_json::json!({"success": true})),
            error: None,
            id: RequestId::Number(1),
        };

        assert!(!response.is_error());
        assert!(response.result().is_some());
    }

    #[test]
    fn test_json_rpc_response_error() {
        let response = JsonRpcResponse {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: None,
            error: Some(JsonRpcError {
                code: error_codes::METHOD_NOT_FOUND,
                message: "Method not found".to_string(),
                data: None,
            }),
            id: RequestId::Number(1),
        };

        assert!(response.is_error());
        assert!(response.result().is_none());
    }

    #[test]
    fn test_mcp_tool_definition_deserialization() {
        let json = r#"{
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }"#;

        let tool: McpToolDefinition = serde_json::from_str(json).expect("deserialize");
        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description.as_deref(), Some("A test tool"));
    }

    #[test]
    fn test_mcp_content_text() {
        let content = McpContent::Text {
            text: "Hello".to_string(),
        };

        let json = serde_json::to_string(&content).expect("serialize");
        assert!(json.contains("text"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_request_id_variants() {
        let num_id = RequestId::Number(42);
        let str_id = RequestId::String("req-1".to_string());

        let json_num = serde_json::to_string(&num_id).expect("serialize");
        let json_str = serde_json::to_string(&str_id).expect("serialize");

        assert_eq!(json_num, "42");
        assert_eq!(json_str, "\"req-1\"");
    }
}
