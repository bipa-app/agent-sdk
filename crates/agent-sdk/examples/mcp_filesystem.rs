//! MCP (Model Context Protocol) example.
//!
//! Connects to an external MCP server over stdio, registers every tool the
//! server advertises with the agent, and asks the model a question that uses
//! those tools. This example uses the reference filesystem server, which is
//! distributed as an npm package and run via `npx`.
//!
//! # Prerequisites
//!
//! - `node` / `npx` on your PATH (the MCP server is a Node package).
//! - The first run downloads `@modelcontextprotocol/server-filesystem`.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example mcp_filesystem
//! ```

use std::sync::Arc;

use agent_sdk::{
    AgentConfig, AgentEvent, AgentInput, AllowAllHooks, CancellationToken, EventStore,
    InMemoryEventStore, InMemoryStore, ThreadId, ToolContext, ToolRegistry, builder,
    mcp::{McpClient, StdioTransport, register_mcp_tools},
    providers::AnthropicProvider,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY environment variable must be set"))?;

    // Expose the current directory to the MCP filesystem server (read/write
    // within the directory you pass as the final argument).
    let workspace = std::env::current_dir()?;
    let workspace = workspace.to_string_lossy().to_string();

    println!("Spawning MCP filesystem server over stdio (workspace: {workspace})...");
    let transport = StdioTransport::spawn(
        "npx",
        &[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            workspace.as_str(),
        ],
    )?;

    // Initialize the MCP connection (performs the handshake + tools/list).
    let client = Arc::new(McpClient::new(transport, "filesystem".to_string()).await?);

    // Bridge every MCP tool into the agent's tool registry.
    let mut tools: ToolRegistry<()> = ToolRegistry::new();
    register_mcp_tools(&mut tools, Arc::clone(&client)).await?;
    println!("Registered {} MCP tool(s):", tools.len());
    for tool in tools.all() {
        println!("  - {}", tool.name_str());
    }
    println!();

    let event_store = Arc::new(InMemoryEventStore::new());
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .tools(tools)
        // Auto-approve tool calls for this demo.
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .config(AgentConfig {
            max_turns: Some(8),
            system_prompt: "You are a helpful assistant with filesystem access via MCP tools."
                .to_string(),
            ..Default::default()
        })
        .event_store(event_store.clone())
        .build_with_stores();

    let thread_id = ThreadId::new();

    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text(
            "List the files in the workspace directory and summarize what this project is."
                .to_string(),
        ),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let _ = final_state.await?;

    for envelope in event_store.get_events(&thread_id).await? {
        match envelope.event {
            AgentEvent::ToolCallStart { name, .. } => {
                println!("[using MCP tool: {name}]");
            }
            AgentEvent::Text { text, .. } => {
                println!("\nAgent: {text}");
            }
            AgentEvent::Done { total_turns, .. } => {
                println!("\n(completed in {total_turns} turns)");
            }
            AgentEvent::Error { message, .. } => {
                eprintln!("error: {message}");
            }
            _ => {}
        }
    }

    Ok(())
}
