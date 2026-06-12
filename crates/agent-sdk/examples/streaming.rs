//! Streaming output example.
//!
//! Prints the agent's reply to stdout token-by-token as it streams, instead of
//! waiting for the whole turn to finish. The trick is to wrap the configured
//! [`EventStore`] so we observe every [`AgentEvent`] the agent loop emits — the
//! agent writes a [`AgentEvent::TextDelta`] for each streamed chunk.
//!
//! Rather than hand-roll the full [`EventStore`] surface, this uses the SDK's
//! reusable [`ObservingEventStore`] decorator: pass an inner store plus a
//! closure that runs on every appended envelope.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example streaming
//! ```

use std::io::Write as _;
use std::sync::Arc;

use agent_sdk::{
    AgentEvent, AgentInput, CancellationToken, EventStore, InMemoryEventStore, ObservingEventStore,
    ThreadId, ToolContext, builder, providers::AnthropicProvider,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY environment variable must be set"))?;

    // Wrap an in-memory store so we observe each streamed delta as it is
    // appended; the inner store still persists every envelope.
    let event_store: Arc<dyn EventStore> = Arc::new(ObservingEventStore::new(
        InMemoryEventStore::new(),
        |envelope| {
            if let AgentEvent::TextDelta { delta, .. } = &envelope.event {
                print!("{delta}");
                let _ = std::io::stdout().flush();
            }
        },
    ));
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .event_store(event_store)
        .build();

    let thread_id = ThreadId::new();

    print!("Agent: ");
    std::io::stdout().flush()?;

    let final_state = agent.run(
        thread_id,
        AgentInput::Text("Write a haiku about the Rust borrow checker.".to_string()),
        ToolContext::new(()),
        CancellationToken::new(),
    );
    let _ = final_state.await?;

    // The streamed deltas above are not newline-terminated.
    println!();

    Ok(())
}
