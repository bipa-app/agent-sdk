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
//! This implementation supports MCP protocol version 2024-11-05 and includes:
//!
//! - JSON-RPC 2.0 communication
//! - Tool discovery via `tools/list`
//! - Tool execution via `tools/call`
//! - Automatic initialization handshake

pub mod client;
pub mod protocol;
pub mod tool_bridge;
pub mod transport;

pub use client::McpClient;
pub use protocol::{
    JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpContent, McpServerCapabilities,
    McpToolCallResult, McpToolDefinition,
};
pub use tool_bridge::{McpToolBridge, register_mcp_tools, register_mcp_tools_with_tiers};
pub use transport::{McpTransport, StdioTransport};
