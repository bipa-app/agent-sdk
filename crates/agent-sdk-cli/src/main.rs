//! `agent-sdk` developer CLI.
//!
//! Today the only subcommand worth shipping is the local Langfuse +
//! `OTel` collector stack used by Phase 9 work. Future verbs (e.g.
//! Grafana, dashboards, eval helpers) plug in alongside it.
//!
//! ```bash
//! agent-sdk run "What is the capital of France?"
//! agent-sdk chat
//! agent-sdk local-langfuse init
//! agent-sdk local-langfuse up
//! agent-sdk doctor
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;
mod embed;

use commands::{agent, doctor, local_langfuse};

#[derive(Parser, Debug)]
#[command(
    name = "agent-sdk",
    version,
    about = "Developer-experience CLI for the Agent SDK",
    long_about = "Materializes the local Langfuse + OTel collector stack into a downstream \
                  consumer's working tree, plus environment sanity checks. Read-only by \
                  default — `up`/`down` shell out to `docker compose` and only fire when \
                  you ask them to.",
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a single prompt against an Anthropic-backed agent and stream the
    /// reply to stdout. Reads `ANTHROPIC_API_KEY` from the environment.
    Run(agent::RunArgs),
    /// Interactive chat with an Anthropic-backed agent; streams replies and
    /// keeps conversation history for the session. Reads `ANTHROPIC_API_KEY`.
    Chat(agent::ChatArgs),
    /// Manage the local Langfuse + `OTel` collector dev stack.
    LocalLangfuse {
        #[command(subcommand)]
        action: local_langfuse::Action,
    },
    /// Check the local environment (docker, ports, dest writability).
    Doctor(doctor::Args),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => agent::run(args),
        Command::Chat(args) => agent::chat(args),
        Command::LocalLangfuse { action } => local_langfuse::run(action),
        Command::Doctor(args) => doctor::run(args),
    }
}
