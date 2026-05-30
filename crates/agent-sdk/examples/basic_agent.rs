//! Basic agent example.
//!
//! This example demonstrates the simplest way to create and run an agent
//! using the high-level [`ask`](agent_sdk::AgentLoop::ask) convenience.
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

use agent_sdk::{ThreadId, builder, providers::AnthropicProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging for debug output (SDK uses the `log` crate)
    env_logger::init();

    // Build the agent. `from_env` reads ANTHROPIC_API_KEY; the builder uses
    // an in-memory event store by default, so there is no Arc/store ceremony.
    let agent = builder::<()>()
        .provider(AnthropicProvider::try_from_env()?)
        .build();

    // Ask a question and print the assembled answer in one call.
    let answer = agent
        .ask(
            ThreadId::new(),
            "What is the capital of France? Answer in one sentence.",
        )
        .await?;

    println!("Agent: {answer}");

    Ok(())
}
