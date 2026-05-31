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
use agent_service_host::grpc::GrpcTransport;
use agent_service_host::host::ServiceHost;
use agent_service_host::runtime::{
    AllowAllConfirmationPolicy, ExecutionRuntime, NoopToolExecutor, StaticProviderResolver,
};
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
        tools_fn: None,
        policy: RuntimePolicy::server_default(),
    };
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(default_definition));
    let runtime = Arc::new(ExecutionRuntime::new(
        Arc::new(StaticProviderResolver::new()),
        Arc::new(NoopToolExecutor),
        Arc::new(AllowAllConfirmationPolicy),
    ));

    let grpc_enabled = config.transport.grpc_enabled;
    let grpc_addr = config.transport.grpc_addr;
    let lease_duration = config.worker.lease_duration();
    let admission = config.admission.clone();
    let host = ServiceHost::new(config, registry, Arc::clone(&runtime))
        .context("creating service host")?;
    let stores = host.stores().clone();
    let health = Arc::clone(host.health());
    let shutdown = host.shutdown_token();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?
        .block_on(async move {
            // Install the global OTel tracer + meter providers as
            // soon as we are inside the runtime. Batch
            // span / periodic metric exporters need a tokio runtime
            // for their flush tasks, so we deliberately stay inside
            // `block_on` rather than installing in `fn main`. The
            // returned guard is dropped at the *end* of this block,
            // so `host.run().await` finishes first and any in-flight
            // exports complete before tear-down.
            //
            // `install_observability` returns `Ok(None)` when
            // `observability.enabled = false` so the default
            // deployment story is unchanged: build without OTel
            // dependencies, run without OTel pipelines.
            #[cfg(feature = "otel")]
            let _otel_guard = agent_service_host::observability::install_observability(
                &host.config().observability,
            )
            .context("installing OTel global provider")?;

            host.initialize()
                .await
                .context("initializing service host")?;
            if grpc_enabled {
                let grpc = GrpcTransport::with_admission(
                    stores,
                    runtime,
                    health,
                    shutdown,
                    lease_duration,
                    admission,
                );
                tokio::try_join!(host.run(), grpc.serve(grpc_addr))?;
                Ok(())
            } else {
                host.run().await
            }
        })
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
