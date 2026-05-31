//! Remote (streamable-HTTP) MCP example.
//!
//! Connects to a *remote* MCP server over the streamable-HTTP transport using
//! an OAuth / bearer token, negotiates a current protocol revision, and lists
//! the server's tools, resources, and prompts. This is the deployment pattern
//! hosted MCP providers use (a single HTTPS endpoint), as opposed to spawning a
//! local subprocess over stdio.
//!
//! # Running
//!
//! ```bash
//! MCP_URL=https://example.com/mcp \
//! MCP_TOKEN=your_oauth_access_token \
//!   cargo run --example mcp_http_remote --features mcp
//! ```

use std::sync::Arc;

use agent_sdk::mcp::{McpAuth, McpClient, StreamableHttpTransport};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let endpoint = std::env::var("MCP_URL")
        .map_err(|_| anyhow::anyhow!("MCP_URL environment variable must be set"))?;

    // Bearer / OAuth access token sent as `Authorization: Bearer <token>`.
    // Use `McpAuth::None` for an unauthenticated endpoint.
    let auth = match std::env::var("MCP_TOKEN") {
        Ok(token) if !token.is_empty() => McpAuth::Bearer(token),
        _ => McpAuth::None,
    };

    println!("Connecting to remote MCP server at {endpoint} (streamable-HTTP)...");
    let transport = StreamableHttpTransport::new(&endpoint, auth)?;
    let client = Arc::new(McpClient::new(transport, "remote".to_string()).await?);

    if let Some(version) = client.protocol_version() {
        println!("Negotiated MCP protocol revision: {version}");
    }

    let tools = client.list_tools().await?;
    println!("\nTools ({}):", tools.len());
    for tool in &tools {
        println!("  - {}", tool.name);
    }

    if client.supports_resources() {
        let resources = client.list_resources().await?;
        println!("\nResources ({}):", resources.len());
        for resource in &resources {
            let name = resource.name.as_deref().unwrap_or("<unnamed>");
            println!("  - {name} ({})", resource.uri);
        }
    } else {
        println!("\nServer does not advertise the resources capability.");
    }

    if client.supports_prompts() {
        let prompts = client.list_prompts().await?;
        println!("\nPrompts ({}):", prompts.len());
        for prompt in &prompts {
            println!("  - {}", prompt.name);
        }
    } else {
        println!("\nServer does not advertise the prompts capability.");
    }

    Ok(())
}
