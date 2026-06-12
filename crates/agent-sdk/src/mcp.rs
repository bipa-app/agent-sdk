//! Model Context Protocol (MCP) client support.
//!
//! This module provides a client for connecting to MCP servers,
//! allowing agents to use tools provided by external services.
//!
//! # Overview
//!
//! MCP (Model Context Protocol) is a protocol for connecting LLM applications
//! to external tools and services. This module provides:
//!
//! - [`McpClient`] - Client for communicating with MCP servers
//! - [`McpTransport`] - Trait for transport implementations
//! - [`StdioTransport`] - Stdio-based transport (subprocess communication)
//! - [`StreamableHttpTransport`] - Streamable-HTTP / SSE transport with OAuth /
//!   bearer auth for remote (hosted) MCP servers
//! - [`McpToolBridge`] - Wrapper to use MCP tools as SDK tools
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::mcp::{McpClient, StdioTransport, register_mcp_tools};
//! use agent_sdk::ToolRegistry;
//! use std::sync::Arc;
//!
//! // Spawn an MCP server process
//! let transport = StdioTransport::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem"]).await?;
//!
//! // Create client and initialize
//! let client = Arc::new(McpClient::new(transport, "filesystem".to_string()).await?);
//!
//! // Register all MCP tools with the agent
//! let mut registry = ToolRegistry::new();
//! register_mcp_tools(&mut registry, client).await?;
//! ```
//!
//! # MCP Protocol
//!
//! This implementation negotiates a current MCP protocol revision during the
//! `initialize` handshake (advertising [`protocol::PREFERRED_PROTOCOL_VERSION`],
//! `2025-06-18`, and honouring whatever revision the server selects — including
//! legacy `2024-11-05` servers). It includes:
//!
//! - JSON-RPC 2.0 communication over stdio or streamable-HTTP / SSE
//! - Protocol-revision negotiation (no longer pinned to `2024-11-05`)
//! - Tool discovery (`tools/list`) and execution (`tools/call`)
//! - Resources surface: `resources/list`, `resources/read`
//! - Prompts surface: `prompts/list`, `prompts/get`
//! - Automatic initialization handshake
//!
//! # Connecting to a remote (HTTP) MCP server
//!
//! ```ignore
//! use agent_sdk::mcp::{McpAuth, McpClient, StreamableHttpTransport};
//! use std::sync::Arc;
//!
//! let transport = StreamableHttpTransport::new(
//!     "https://example.com/mcp",
//!     McpAuth::Bearer(std::env::var("MCP_TOKEN")?),
//! )?;
//! let client = Arc::new(McpClient::new(transport, "remote".to_string()).await?);
//! let tools = client.list_tools().await?;
//! ```

pub mod client;
pub mod http;
pub mod protocol;
pub mod tool_bridge;
pub mod transport;

pub use client::McpClient;
pub use http::{
    DEFAULT_HTTP_TIMEOUT, DEFAULT_SEND_DEADLINE, HttpPoster, HttpReply, HttpRequest, McpAuth,
    ReqwestPoster, StreamableHttpTransport,
};
pub use protocol::{
    JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpContent, McpPrompt, McpPromptArgument,
    McpPromptMessage, McpResource, McpResourceContents, McpRole, McpServerCapabilities,
    McpToolCallResult, McpToolDefinition, PromptGetResult, ResourceReadResult,
};
pub use tool_bridge::{McpToolBridge, register_mcp_tools, register_mcp_tools_with_tiers};
pub use transport::{McpTransport, StdioTransport};
