//! Basic agent example.
//!
//! This example demonstrates the simplest way to create and run an agent.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example basic_agent
//! ```
//!
//! To see debug logs from the SDK:
//! ```bash
//! RUST_LOG=agent_sdk=debug ANTHROPIC_API_KEY=your_key cargo run --example basic_agent
//! ```

use agent_sdk::{
    AgentEvent, AgentInput, CancellationToken, EventStore, InMemoryEventStore, ThreadId,
    ToolContext, builder, providers::AnthropicProvider,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging for debug output (SDK uses the `log` crate)
    env_logger::init();

    // Get API key from environment
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    // Build the agent with default settings
    let event_store = Arc::new(InMemoryEventStore::new());
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .event_store(event_store.clone())
        .build();

    // Create a new conversation thread
    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());

    println!("Starting conversation (thread: {thread_id})\n");

    // Run the agent with a simple prompt
    let final_state = agent.run(
        thread_id.clone(),
        AgentInput::Text("What is the capital of France? Answer in one sentence.".to_string()),
        tool_ctx,
        CancellationToken::new(),
    );
    let _ = final_state.await?;

    for envelope in event_store.get_events(&thread_id).await? {
        match envelope.event {
            AgentEvent::Text {
                message_id: _,
                text,
            } => {
                println!("Agent: {text}");
            }
            AgentEvent::Done {
                total_turns,
                total_usage,
                duration,
                ..
            } => {
                println!("\n---");
                println!(
                    "Completed in {} turns, {:.2}s",
                    total_turns,
                    duration.as_secs_f64()
                );
                println!(
                    "Tokens: {} input, {} output",
                    total_usage.input_tokens, total_usage.output_tokens
                );
            }
            AgentEvent::Error { message, .. } => {
                eprintln!("Error: {message}");
            }
            _ => {}
        }
    }

    Ok(())
}
