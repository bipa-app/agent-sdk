//! MCP JSON-RPC protocol types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC version string.
pub const JSONRPC_VERSION: &str = "2.0";

/// The MCP protocol revision this client prefers to negotiate.
///
/// MCP revisions are date-stamped. `2025-06-18` is the current stable
/// revision and supersedes the original `2024-11-05` this client was
/// previously pinned to. During the `initialize` handshake we advertise
/// this version; the server replies with the revision it actually selected
/// (which may be older), and we honour that for subsequent requests
/// (notably the `MCP-Protocol-Version` HTTP header).
pub const PREFERRED_PROTOCOL_VERSION: &str = "2025-06-18";

/// The oldest MCP revision this client interoperates with.
///
/// Servers that only speak `2024-11-05` are still supported: the client
/// adapts to whatever revision the server selects during initialization.
pub const MIN_PROTOCOL_VERSION: &str = "2024-11-05";

/// MCP protocol revisions this client knows about, newest first.
///
/// Used to decide whether a server-selected revision is one we recognise.
/// An unknown revision is not fatal — the client still proceeds — but it is
/// logged so operators can tell when a server negotiated something outside
/// this set.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Returns `true` if `version` is a revision this client explicitly knows.
#[must_use]
pub fn is_known_protocol_version(version: &str) -> bool {
    SUPPORTED_PROTOCOL_VERSIONS.contains(&version)
}

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

// ── Resources ──────────────────────────────────────────────────────────

/// An MCP resource descriptor as returned by `resources/list`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpResource {
    /// Resource URI (used to read the resource via `resources/read`).
    pub uri: String,
    /// Human-readable resource name.
    #[serde(default)]
    pub name: Option<String>,
    /// Optional resource description.
    #[serde(default)]
    pub description: Option<String>,
    /// MIME type of the resource contents, if known.
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
}

/// `resources/list` response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourcesListResult {
    /// Available resources.
    pub resources: Vec<McpResource>,
    /// Opaque cursor for pagination, if the server returned more pages.
    #[serde(default, rename = "nextCursor")]
    pub next_cursor: Option<String>,
}

/// `resources/read` params.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceReadParams {
    /// URI of the resource to read.
    pub uri: String,
}

/// The contents of a single resource returned by `resources/read`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpResourceContents {
    /// Resource URI.
    pub uri: String,
    /// MIME type of the contents, if known.
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
    /// Text contents (mutually exclusive with `blob`).
    #[serde(default)]
    pub text: Option<String>,
    /// Base64-encoded binary contents (mutually exclusive with `text`).
    #[serde(default)]
    pub blob: Option<String>,
}

/// `resources/read` response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceReadResult {
    /// One or more content blocks for the requested resource.
    pub contents: Vec<McpResourceContents>,
}

// ── Prompts ────────────────────────────────────────────────────────────

/// A prompt argument descriptor from `prompts/list`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpPromptArgument {
    /// Argument name.
    pub name: String,
    /// Optional argument description.
    #[serde(default)]
    pub description: Option<String>,
    /// Whether the argument is required.
    #[serde(default)]
    pub required: bool,
}

/// An MCP prompt descriptor as returned by `prompts/list`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpPrompt {
    /// Prompt name (used to fetch the prompt via `prompts/get`).
    pub name: String,
    /// Optional prompt description.
    #[serde(default)]
    pub description: Option<String>,
    /// Declared prompt arguments.
    #[serde(default)]
    pub arguments: Vec<McpPromptArgument>,
}

/// `prompts/list` response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptsListResult {
    /// Available prompts.
    pub prompts: Vec<McpPrompt>,
    /// Opaque cursor for pagination, if the server returned more pages.
    #[serde(default, rename = "nextCursor")]
    pub next_cursor: Option<String>,
}

/// `prompts/get` params.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptGetParams {
    /// Name of the prompt to fetch.
    pub name: String,
    /// Arguments to interpolate into the prompt template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

/// The role of a prompt message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpRole {
    /// A user-authored message.
    User,
    /// An assistant-authored message.
    Assistant,
}

/// A single message in a rendered prompt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpPromptMessage {
    /// Message role.
    pub role: McpRole,
    /// Message content block.
    pub content: McpContent,
}

/// `prompts/get` response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptGetResult {
    /// Optional prompt description.
    #[serde(default)]
    pub description: Option<String>,
    /// The rendered prompt messages.
    pub messages: Vec<McpPromptMessage>,
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

    #[test]
    fn preferred_protocol_version_is_newer_than_floor() {
        // The preferred revision must be recognised and must not be the
        // legacy floor — otherwise we never moved off `2024-11-05`.
        assert_ne!(PREFERRED_PROTOCOL_VERSION, MIN_PROTOCOL_VERSION);
        assert!(is_known_protocol_version(PREFERRED_PROTOCOL_VERSION));
        assert!(is_known_protocol_version(MIN_PROTOCOL_VERSION));
        assert!(!is_known_protocol_version("1999-01-01"));
        // Newest-first ordering: the preferred revision leads the list.
        assert_eq!(SUPPORTED_PROTOCOL_VERSIONS[0], PREFERRED_PROTOCOL_VERSION);
    }

    #[test]
    fn test_resources_list_deserialization() {
        let json = r#"{
            "resources": [
                {"uri": "file:///a.txt", "name": "A", "mimeType": "text/plain"},
                {"uri": "mem://b"}
            ],
            "nextCursor": "page2"
        }"#;
        let parsed: ResourcesListResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.resources.len(), 2);
        assert_eq!(parsed.resources[0].uri, "file:///a.txt");
        assert_eq!(parsed.resources[0].mime_type.as_deref(), Some("text/plain"));
        assert_eq!(parsed.resources[1].name, None);
        assert_eq!(parsed.next_cursor.as_deref(), Some("page2"));
    }

    #[test]
    fn test_resource_read_text_and_blob() {
        let json = r#"{
            "contents": [
                {"uri": "file:///a.txt", "mimeType": "text/plain", "text": "hello"},
                {"uri": "file:///b.bin", "blob": "AAAA"}
            ]
        }"#;
        let parsed: ResourceReadResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.contents.len(), 2);
        assert_eq!(parsed.contents[0].text.as_deref(), Some("hello"));
        assert_eq!(parsed.contents[1].blob.as_deref(), Some("AAAA"));
        assert_eq!(parsed.contents[1].text, None);
    }

    #[test]
    fn test_prompts_list_deserialization() {
        let json = r#"{
            "prompts": [
                {
                    "name": "summarize",
                    "description": "Summarize text",
                    "arguments": [
                        {"name": "text", "required": true},
                        {"name": "tone", "description": "voice", "required": false}
                    ]
                }
            ]
        }"#;
        let parsed: PromptsListResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.prompts.len(), 1);
        assert_eq!(parsed.prompts[0].name, "summarize");
        assert_eq!(parsed.prompts[0].arguments.len(), 2);
        assert!(parsed.prompts[0].arguments[0].required);
        assert!(!parsed.prompts[0].arguments[1].required);
    }

    #[test]
    fn test_prompt_get_messages() {
        let json = r#"{
            "description": "rendered",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": "hi"}},
                {"role": "assistant", "content": {"type": "text", "text": "hello"}}
            ]
        }"#;
        let parsed: PromptGetResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.messages.len(), 2);
        assert_eq!(parsed.messages[0].role, McpRole::User);
        assert_eq!(parsed.messages[1].role, McpRole::Assistant);
        match &parsed.messages[0].content {
            McpContent::Text { text } => assert_eq!(text, "hi"),
            _ => panic!("expected text content"),
        }
    }
}
