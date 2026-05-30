//! Streaming output example.
//!
//! Prints the agent's reply to stdout token-by-token as it streams, instead of
//! waiting for the whole turn to finish. The trick is to wrap the configured
//! [`EventStore`] so we observe every [`AgentEvent`] the agent loop emits — the
//! agent writes a [`AgentEvent::TextDelta`] for each streamed chunk.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example streaming
//! ```

use std::io::Write as _;
use std::sync::Arc;

use agent_sdk::{
    AgentEvent, AgentEventEnvelope, AgentInput, CancellationToken, EventStore, InMemoryEventStore,
    StoredTurnEvents, ThreadId, ToolContext, builder, providers::AnthropicProvider,
};
use async_trait::async_trait;

/// Wraps an [`InMemoryEventStore`], printing streamed text as it arrives and
/// delegating all persistence to the inner store.
struct PrintingEventStore {
    inner: Arc<InMemoryEventStore>,
}

impl PrintingEventStore {
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryEventStore::new()),
        }
    }
}

#[async_trait]
impl EventStore for PrintingEventStore {
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> anyhow::Result<()> {
        if let AgentEvent::TextDelta { delta, .. } = &envelope.event {
            print!("{delta}");
            let _ = std::io::stdout().flush();
        }
        self.inner.append(thread_id, turn, envelope).await
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> anyhow::Result<()> {
        self.inner.finish_turn(thread_id, turn).await
    }

    async fn get_turn(
        &self,
        thread_id: &ThreadId,
        turn: usize,
    ) -> anyhow::Result<Option<StoredTurnEvents>> {
        self.inner.get_turn(thread_id, turn).await
    }

    async fn get_turns(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<StoredTurnEvents>> {
        self.inner.get_turns(thread_id).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> anyhow::Result<()> {
        self.inner.clear(thread_id).await
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY environment variable must be set"))?;

    let event_store = Arc::new(PrintingEventStore::new());
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
