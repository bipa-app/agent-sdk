//! `agent-sdk run` / `agent-sdk chat` — talk to an Anthropic-backed agent.
//!
//! Both subcommands build a real [`agent_sdk`] agent, read the API key from
//! the `ANTHROPIC_API_KEY` environment variable, and stream the model's
//! response to stdout as it arrives.
//!
//! - `run "<prompt>"` runs a single prompt and exits.
//! - `chat` opens an interactive REPL; conversation history is kept for the
//!   life of the process so the agent remembers earlier turns.
//!
//! ```bash
//! ANTHROPIC_API_KEY=sk-... agent-sdk run "Explain Rust ownership in two sentences."
//! ANTHROPIC_API_KEY=sk-... agent-sdk chat
//! ```

use std::io::Write as _;
use std::sync::Arc;

use agent_sdk::{
    AgentConfig, AgentEvent, AgentEventEnvelope, AgentInput, CancellationToken, EventStore,
    InMemoryEventStore, ThreadId, ToolContext, builder, providers::AnthropicProvider,
};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use clap::Args as ClapArgs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// `agent-sdk run "<prompt>"`.
#[derive(ClapArgs, Debug)]
pub struct RunArgs {
    /// The prompt to send to the agent.
    pub prompt: String,

    /// System prompt to steer the agent.
    #[arg(long, default_value = "You are a helpful assistant.")]
    pub system: String,

    /// Anthropic model alias to use.
    #[arg(long, value_enum, default_value_t = Model::Sonnet)]
    pub model: Model,
}

/// `agent-sdk chat`.
#[derive(ClapArgs, Debug)]
pub struct ChatArgs {
    /// System prompt to steer the agent.
    #[arg(long, default_value = "You are a helpful assistant.")]
    pub system: String,

    /// Anthropic model alias to use.
    #[arg(long, value_enum, default_value_t = Model::Sonnet)]
    pub model: Model,
}

/// Anthropic model aliases exposed by the CLI.
#[derive(Copy, Clone, Debug, clap::ValueEnum)]
pub enum Model {
    /// Claude Haiku — fastest, cheapest.
    Haiku,
    /// Claude Sonnet — balanced (default).
    Sonnet,
    /// Claude Opus — most capable.
    Opus,
}

impl Model {
    fn provider(self, api_key: String) -> AnthropicProvider {
        match self {
            Self::Haiku => AnthropicProvider::haiku(api_key),
            Self::Sonnet => AnthropicProvider::sonnet(api_key),
            Self::Opus => AnthropicProvider::opus(api_key),
        }
    }
}

/// An [`EventStore`] decorator that prints streaming text to stdout as the
/// agent produces it, then delegates persistence to an inner store.
///
/// This is the idiomatic way to "stream to stdout" with the public SDK: the
/// agent loop writes every [`AgentEvent`] through the configured event store,
/// so wrapping that store lets us observe events live without an in-process
/// channel.
struct StreamToStdout {
    inner: Arc<InMemoryEventStore>,
}

impl StreamToStdout {
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryEventStore::new()),
        }
    }
}

#[async_trait]
impl EventStore for StreamToStdout {
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> Result<()> {
        match &envelope.event {
            AgentEvent::TextDelta { delta, .. } => {
                print!("{delta}");
                // Deltas arrive mid-line; flush so they show up immediately.
                let _ = std::io::stdout().flush();
            }
            AgentEvent::Error { message, .. } => {
                eprintln!("\nerror: {message}");
            }
            _ => {}
        }
        self.inner.append(thread_id, turn, envelope).await
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> Result<()> {
        self.inner.finish_turn(thread_id, turn).await
    }

    async fn get_turn(
        &self,
        thread_id: &ThreadId,
        turn: usize,
    ) -> Result<Option<agent_sdk::StoredTurnEvents>> {
        self.inner.get_turn(thread_id, turn).await
    }

    async fn get_turns(&self, thread_id: &ThreadId) -> Result<Vec<agent_sdk::StoredTurnEvents>> {
        self.inner.get_turns(thread_id).await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        self.inner.clear(thread_id).await
    }
}

fn require_api_key() -> Result<String> {
    let key = std::env::var(API_KEY_ENV).with_context(|| {
        format!("{API_KEY_ENV} is not set; export your Anthropic API key to run an agent")
    })?;
    if key.trim().is_empty() {
        bail!("{API_KEY_ENV} is set but empty");
    }
    Ok(key)
}

/// Entry point for `agent-sdk run`.
///
/// # Errors
/// Returns an error if the API key is missing or the agent run fails.
pub fn run(args: RunArgs) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new().context("failed to start async runtime")?;
    runtime.block_on(run_async(args))
}

/// Entry point for `agent-sdk chat`.
///
/// # Errors
/// Returns an error if the API key is missing or the agent run fails.
pub fn chat(args: ChatArgs) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new().context("failed to start async runtime")?;
    runtime.block_on(chat_async(args))
}

async fn run_async(args: RunArgs) -> Result<()> {
    let api_key = require_api_key()?;
    let agent = build_agent(args.model.provider(api_key), args.system);

    let thread_id = ThreadId::new();
    let _ = agent
        .run(
            thread_id,
            AgentInput::Text(args.prompt),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        .context("agent run failed")?;

    // The streamed deltas above are not newline-terminated.
    println!();
    Ok(())
}

async fn chat_async(args: ChatArgs) -> Result<()> {
    let api_key = require_api_key()?;
    let agent = build_agent(args.model.provider(api_key), args.system);

    // One thread for the whole session keeps conversation history in the
    // agent's message store, so each turn sees the earlier ones.
    let thread_id = ThreadId::new();

    let mut stdout = tokio::io::stdout();
    stdout
        .write_all(b"agent-sdk chat - type a message, or 'exit' / Ctrl-D to quit.\n")
        .await?;
    stdout.flush().await?;

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        stdout.write_all(b"\nyou> ").await?;
        stdout.flush().await?;

        let Some(line) = lines.next_line().await? else {
            // EOF (Ctrl-D).
            stdout.write_all(b"\n").await?;
            break;
        };
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if matches!(prompt, "exit" | "quit") {
            break;
        }

        stdout.write_all(b"\nagent> ").await?;
        stdout.flush().await?;

        let _ = agent
            .run(
                thread_id.clone(),
                AgentInput::Text(prompt.to_string()),
                ToolContext::new(()),
                CancellationToken::new(),
            )
            .await
            .context("agent run failed")?;

        // Terminate the streamed (delta) line.
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }

    Ok(())
}

type CliAgent = agent_sdk::AgentLoop<
    (),
    AnthropicProvider,
    agent_sdk::DefaultHooks,
    agent_sdk::InMemoryStore,
    agent_sdk::InMemoryStore,
>;

fn build_agent(provider: AnthropicProvider, system: String) -> CliAgent {
    let event_store = Arc::new(StreamToStdout::new());
    builder::<()>()
        .provider(provider)
        .config(AgentConfig {
            system_prompt: system,
            ..Default::default()
        })
        .event_store(event_store)
        .build()
}
