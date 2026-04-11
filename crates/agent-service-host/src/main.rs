//! Binary entry point for the agent service host.
//!
//! Loads configuration, initialises tracing, constructs the
//! [`ServiceHost`], and runs it until a shutdown signal arrives.
//!
//! # Configuration
//!
//! The config path is resolved in order:
//!
//! 1. `AGENT_SERVICE_CONFIG` environment variable — **must** point to
//!    an existing file or the process exits with an error.
//! 2. `./config.yaml` in the working directory (if it exists).
//! 3. If neither is present, the built-in defaults are used.
//!
//! [`ServiceHost`]: agent_service_host::host::ServiceHost

use std::path::PathBuf;
use std::sync::Arc;

use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
use agent_service_host::config::ServiceConfig;
use agent_service_host::host::ServiceHost;
use anyhow::{Context, Result};
use tracing::info;

fn main() -> Result<()> {
    init_tracing();

    let config = load_config().context("loading configuration")?;
    info!(?config, "configuration loaded");

    // Phase 1: use the in-memory definition registry with a default
    // definition. A future phase will load definitions from config or
    // a database.
    let default_definition = AgentDefinition {
        provider: "anthropic".into(),
        model: "claude-sonnet-4-5-20250929".into(),
        system_prompt: String::new(),
        max_tokens: 4096,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        policy: RuntimePolicy::server_default(),
    };
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(default_definition));

    let host = ServiceHost::new(config, registry).context("creating service host")?;

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?
        .block_on(host.run())
}

/// Resolve and load configuration.
fn load_config() -> Result<ServiceConfig> {
    config_path()?.map_or_else(
        || {
            info!("no config path specified, using defaults");
            Ok(ServiceConfig::default())
        },
        |p| {
            info!(path = %p.display(), "loading config from file");
            ServiceConfig::from_yaml_file(&p)
        },
    )
}

/// Determine the config file path.
///
/// Returns `Err` if `AGENT_SERVICE_CONFIG` is set but the file does
/// not exist — this is an explicit operator intent that must not be
/// silently ignored.
fn config_path() -> Result<Option<PathBuf>> {
    if let Ok(env_path) = std::env::var("AGENT_SERVICE_CONFIG") {
        let p = PathBuf::from(&env_path);
        anyhow::ensure!(p.exists(), "AGENT_SERVICE_CONFIG={env_path} does not exist");
        return Ok(Some(p));
    }
    let default = PathBuf::from("config.yaml");
    if default.exists() {
        return Ok(Some(default));
    }
    Ok(None)
}

/// Initialise the `tracing` subscriber.
fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt;

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,agent_service_host=debug,agent_server=debug"));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .init();
}
