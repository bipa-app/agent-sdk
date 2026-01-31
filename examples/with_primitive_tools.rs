//! Using primitive tools example.
//!
//! This example shows how to set up an agent with the SDK's built-in
//! primitive tools (Read, Write, Edit, Grep, Glob, Bash).
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example with_primitive_tools
//! ```
//!
//! To see debug logs from the SDK:
//! ```bash
//! RUST_LOG=agent_sdk=debug ANTHROPIC_API_KEY=your_key cargo run --example with_primitive_tools
//! ```

use agent_sdk::{
    AgentCapabilities, AgentConfig, AgentEvent, AgentInput, AllowAllHooks, Environment,
    InMemoryFileSystem, InMemoryStore, ThreadId, ToolContext, ToolRegistry, builder,
    primitive_tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool},
    providers::AnthropicProvider,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    // Create an in-memory filesystem for this example
    // In production, you'd use LocalFileSystem
    let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

    // Pre-populate with some files
    fs.write_file("README.md", "# My Project\n\nThis is a test project.")
        .await?;
    fs.write_file(
        "src/main.rs",
        r#"fn main() {
    println!("Hello, world!");
}
"#,
    )
    .await?;
    fs.write_file(
        "src/lib.rs",
        r#"pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
"#,
    )
    .await?;

    println!("Created in-memory workspace with files:");
    println!("  /workspace/README.md");
    println!("  /workspace/src/main.rs");
    println!("  /workspace/src/lib.rs");
    println!();

    // Configure capabilities (what the agent can do)
    // For safety, we use read-only + write in this example (no exec)
    let capabilities = AgentCapabilities::read_only().with_write(true);

    // Register the primitive tools
    let mut tools: ToolRegistry<()> = ToolRegistry::new();
    tools
        .register(ReadTool::new(Arc::clone(&fs), capabilities.clone()))
        .register(WriteTool::new(Arc::clone(&fs), capabilities.clone()))
        .register(EditTool::new(Arc::clone(&fs), capabilities.clone()))
        .register(GrepTool::new(Arc::clone(&fs), capabilities.clone()))
        .register(GlobTool::new(Arc::clone(&fs), capabilities.clone()))
        .register(BashTool::new(Arc::clone(&fs), capabilities));

    println!("Registered {} primitive tools\n", tools.len());

    // Configure the agent
    let config = AgentConfig {
        max_turns: 10,
        system_prompt: "You are a helpful coding assistant. You have access to file tools. \
                        The workspace is at /workspace."
            .to_string(),
        ..Default::default()
    };

    // Build the agent
    // We use AllowAllHooks to auto-approve tool calls for this demo
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .tools(tools)
        .hooks(AllowAllHooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .config(config)
        .build_with_stores();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());

    // Ask the agent to explore and modify files
    let (mut events, _final_state) = agent.run(
        thread_id,
        AgentInput::Text("List all .rs files in the workspace, then add a simple test to src/lib.rs that tests the greet function.".to_string()),
        tool_ctx,
    );

    println!("--- Agent Working ---\n");

    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::ToolCallStart { name, .. } => {
                println!("[Using tool: {name}]");
            }
            AgentEvent::ToolCallEnd { name, result, .. } => {
                if result.success {
                    // Truncate long outputs
                    let output = if result.output.len() > 200 {
                        format!("{}...", &result.output[..200])
                    } else {
                        result.output.clone()
                    };
                    println!("[{name} result]: {output}\n");
                } else {
                    println!("[{name} failed]: {}\n", result.output);
                }
            }
            AgentEvent::Text { text } => {
                println!("Agent: {text}\n");
            }
            AgentEvent::Done { total_turns, .. } => {
                println!("--- Completed in {total_turns} turns ---\n");
            }
            AgentEvent::Error { message, .. } => {
                eprintln!("Error: {message}");
            }
            _ => {}
        }
    }

    // Show the final state of the modified file
    println!("--- Final src/lib.rs ---");
    let final_content = fs.read_file("/workspace/src/lib.rs").await?;
    println!("{final_content}");

    Ok(())
}
