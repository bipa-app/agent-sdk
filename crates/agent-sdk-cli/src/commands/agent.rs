//! `agent-sdk run` / `agent-sdk chat` — talk to any SDK-backed LLM provider.
//!
//! Both subcommands build a real [`agent_sdk`] agent, select an LLM provider
//! from `--provider` (default `anthropic`), read that provider's credentials
//! from the environment, and stream the model's response to stdout as it
//! arrives.
//!
//! - `run "<prompt>"` runs a single prompt and exits.
//! - `chat` opens an interactive REPL; conversation history is kept for the
//!   life of the process so the agent remembers earlier turns.
//!
//! ```bash
//! ANTHROPIC_API_KEY=sk-... agent-sdk run "Explain Rust ownership in two sentences."
//! OPENAI_API_KEY=sk-...    agent-sdk run --provider openai "Hello"
//! GEMINI_API_KEY=...       agent-sdk chat --provider gemini
//! ```
//!
//! ## Providers and credentials
//!
//! | `--provider` | Credentials (environment / flags)                                   |
//! |--------------|---------------------------------------------------------------------|
//! | `anthropic`  | `ANTHROPIC_API_KEY`                                                  |
//! | `openai`     | `OPENAI_API_KEY`                                                     |
//! | `gemini`     | `GEMINI_API_KEY` or `GOOGLE_API_KEY`                                 |
//! | `vertex`     | `VERTEX_ACCESS_TOKEN`, `GOOGLE_CLOUD_PROJECT`/`--gcp-project`, region |
//! | `cloudflare` | account id + gateway id + upstream key/gateway token (see below)     |

use std::io::Write as _;
use std::sync::Arc;

use agent_sdk::{
    AgentConfig, AgentEvent, AgentEventEnvelope, AgentInput, CancellationToken, EventStore,
    InMemoryEventStore, LlmProvider, ThreadId, ToolContext, builder,
    providers::{
        AnthropicProvider, CloudflareAIGatewayProvider, GeminiProvider, OpenAIProvider,
        VertexProvider,
    },
};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use clap::Args as ClapArgs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Default Anthropic model when `--model` is omitted (Claude Sonnet).
const DEFAULT_ANTHROPIC_MODEL: &str = "sonnet";
/// Default `OpenAI` model when `--model` is omitted (GPT-5.4).
const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4";
/// Default Gemini / Vertex model when `--model` is omitted.
const DEFAULT_GEMINI_MODEL: &str = "gemini-3-flash-preview";
/// Default Vertex region when `--gcp-region` / `VERTEX_REGION` are unset.
const DEFAULT_VERTEX_REGION: &str = "global";

/// LLM provider backends the CLI can drive.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Provider {
    /// Anthropic Messages API. Reads `ANTHROPIC_API_KEY`.
    #[default]
    Anthropic,
    /// `OpenAI` Chat Completions API. Reads `OPENAI_API_KEY`.
    Openai,
    /// Google Gemini API. Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
    Gemini,
    /// Google Vertex AI. Reads `VERTEX_ACCESS_TOKEN` + GCP project/region.
    Vertex,
    /// Cloudflare AI Gateway proxy in front of an upstream provider.
    Cloudflare,
}

/// Upstream provider routed through the Cloudflare AI Gateway.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum CloudflareUpstream {
    /// Route Anthropic Claude models through the gateway.
    #[default]
    Anthropic,
    /// Route `OpenAI` models through the gateway.
    Openai,
    /// Route Gemini models through the gateway.
    Gemini,
}

/// Provider-selection options shared by `run` and `chat`.
#[derive(ClapArgs, Debug, Clone)]
pub struct ProviderArgs {
    /// LLM provider backend to use.
    #[arg(long, value_enum, default_value_t = Provider::Anthropic)]
    pub provider: Provider,

    /// Model to use. Provider-aware: for `anthropic` the aliases
    /// `haiku`/`sonnet`/`opus`/`fable` are accepted in addition to full model
    /// ids; other providers take a full model string. Defaults per provider.
    #[arg(long)]
    pub model: Option<String>,

    /// Cloudflare: upstream provider routed through the gateway.
    #[arg(long, value_enum, default_value_t = CloudflareUpstream::Anthropic)]
    pub cf_upstream: CloudflareUpstream,

    /// Cloudflare account id (or `CLOUDFLARE_ACCOUNT_ID`).
    #[arg(long)]
    pub cf_account_id: Option<String>,

    /// Cloudflare AI Gateway id (or `CLOUDFLARE_GATEWAY_ID`).
    #[arg(long)]
    pub cf_gateway_id: Option<String>,

    /// Vertex/GCP project id (or `GOOGLE_CLOUD_PROJECT`).
    #[arg(long)]
    pub gcp_project: Option<String>,

    /// Vertex/GCP region (or `VERTEX_REGION`; defaults to `global`).
    #[arg(long)]
    pub gcp_region: Option<String>,
}

/// `agent-sdk run "<prompt>"`.
#[derive(ClapArgs, Debug)]
pub struct RunArgs {
    /// The prompt to send to the agent.
    pub prompt: String,

    /// System prompt to steer the agent.
    #[arg(long, default_value = "You are a helpful assistant.")]
    pub system: String,

    #[command(flatten)]
    pub provider: ProviderArgs,
}

/// `agent-sdk chat`.
#[derive(ClapArgs, Debug)]
pub struct ChatArgs {
    /// System prompt to steer the agent.
    #[arg(long, default_value = "You are a helpful assistant.")]
    pub system: String,

    #[command(flatten)]
    pub provider: ProviderArgs,
}

/// Read a required environment variable, failing with an actionable message.
fn require_env(name: &str, hint: &str) -> Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is not set; {hint}"))?;
    if value.trim().is_empty() {
        bail!("{name} is set but empty; {hint}");
    }
    Ok(value)
}

/// Read the first present, non-empty environment variable from `names`.
fn first_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}

/// Resolve an Anthropic model alias or full id into a concrete provider.
fn anthropic_provider(api_key: String, model: &str) -> AnthropicProvider {
    match model {
        "haiku" => AnthropicProvider::haiku(api_key),
        "sonnet" => AnthropicProvider::sonnet(api_key),
        "opus" => AnthropicProvider::opus(api_key),
        "fable" => AnthropicProvider::fable(api_key),
        other => AnthropicProvider::new(api_key, other.to_owned()),
    }
}

/// Resolve a value from an explicit flag, falling back to environment
/// variables, with an actionable error when neither is present.
fn resolve_with_env(flag: Option<&str>, env_names: &[&str], missing: &str) -> Result<String> {
    flag.map(str::to_owned)
        .filter(|value| !value.trim().is_empty())
        .or_else(|| first_env(env_names))
        .context(missing.to_owned())
}

fn build_cloudflare(args: &ProviderArgs) -> Result<CloudflareAIGatewayProvider> {
    let account_id = resolve_with_env(
        args.cf_account_id.as_deref(),
        &["CLOUDFLARE_ACCOUNT_ID"],
        "Cloudflare account id missing; pass --cf-account-id or set CLOUDFLARE_ACCOUNT_ID",
    )?;
    let gateway_id = resolve_with_env(
        args.cf_gateway_id.as_deref(),
        &["CLOUDFLARE_GATEWAY_ID"],
        "Cloudflare gateway id missing; pass --cf-gateway-id or set CLOUDFLARE_GATEWAY_ID",
    )?;

    // BYOK mode uses a gateway token; pass-through mode uses the upstream
    // provider's API key. At least one must be present.
    let gateway_token = first_env(&["CLOUDFLARE_GATEWAY_TOKEN", "CLOUDFLARE_AIG_TOKEN"]);

    let provider = match args.cf_upstream {
        CloudflareUpstream::Anthropic => {
            let model = args
                .model
                .clone()
                .unwrap_or_else(|| "claude-sonnet-4-6".to_owned());
            let api_key = first_env(&["ANTHROPIC_API_KEY"]).unwrap_or_default();
            ensure_cf_auth(&api_key, gateway_token.as_deref(), "ANTHROPIC_API_KEY")?;
            let base =
                CloudflareAIGatewayProvider::anthropic(api_key, &account_id, &gateway_id, model);
            apply_gateway_token(base, gateway_token.as_deref())
        }
        CloudflareUpstream::Openai => {
            let model = args
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned());
            let api_key = first_env(&["OPENAI_API_KEY"]).unwrap_or_default();
            ensure_cf_auth(&api_key, gateway_token.as_deref(), "OPENAI_API_KEY")?;
            let base =
                CloudflareAIGatewayProvider::openai(api_key, &account_id, &gateway_id, model);
            apply_gateway_token(base, gateway_token.as_deref())
        }
        CloudflareUpstream::Gemini => {
            let model = args
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_GEMINI_MODEL.to_owned());
            let api_key = first_env(&["GEMINI_API_KEY", "GOOGLE_API_KEY"]).unwrap_or_default();
            ensure_cf_auth(
                &api_key,
                gateway_token.as_deref(),
                "GEMINI_API_KEY/GOOGLE_API_KEY",
            )?;
            let base =
                CloudflareAIGatewayProvider::gemini(api_key, &account_id, &gateway_id, model);
            apply_gateway_token(base, gateway_token.as_deref())
        }
    };
    Ok(provider)
}

/// Ensure a Cloudflare request has *some* credential: either an upstream
/// provider key (pass-through) or a gateway token (BYOK).
fn ensure_cf_auth(api_key: &str, gateway_token: Option<&str>, key_env: &str) -> Result<()> {
    if api_key.trim().is_empty() && gateway_token.is_none() {
        bail!(
            "Cloudflare requires credentials: set {key_env} for pass-through mode, \
             or CLOUDFLARE_GATEWAY_TOKEN for BYOK mode"
        );
    }
    Ok(())
}

fn apply_gateway_token(
    provider: CloudflareAIGatewayProvider,
    token: Option<&str>,
) -> CloudflareAIGatewayProvider {
    match token {
        Some(token) => provider.with_gateway_token(token),
        None => provider,
    }
}

fn build_vertex(args: &ProviderArgs) -> Result<VertexProvider> {
    let access_token = require_env(
        "VERTEX_ACCESS_TOKEN",
        "export an OAuth2 access token (e.g. `gcloud auth print-access-token`)",
    )?;
    let project_id = resolve_with_env(
        args.gcp_project.as_deref(),
        &["GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"],
        "Vertex project id missing; pass --gcp-project or set GOOGLE_CLOUD_PROJECT",
    )?;
    let region = args
        .gcp_region
        .clone()
        .or_else(|| first_env(&["VERTEX_REGION", "GOOGLE_CLOUD_REGION"]))
        .unwrap_or_else(|| DEFAULT_VERTEX_REGION.to_owned());
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| DEFAULT_GEMINI_MODEL.to_owned());
    Ok(VertexProvider::new(access_token, project_id, region, model))
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

/// Entry point for `agent-sdk run`.
///
/// # Errors
/// Returns an error if credentials are missing or the agent run fails.
pub fn run(args: RunArgs) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new().context("failed to start async runtime")?;
    runtime.block_on(run_async(args))
}

/// Entry point for `agent-sdk chat`.
///
/// # Errors
/// Returns an error if credentials are missing or the agent run fails.
pub fn chat(args: ChatArgs) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new().context("failed to start async runtime")?;
    runtime.block_on(chat_async(args))
}

async fn run_async(args: RunArgs) -> Result<()> {
    let RunArgs {
        prompt,
        system,
        provider,
    } = args;

    // `match` on the provider enum so each arm monomorphizes the generic
    // helper for its concrete provider type. The builder requires a concrete
    // `P: LlmProvider` (there is no `Arc<dyn LlmProvider>` path on this
    // branch), so this is the clean way to support runtime selection.
    match provider.provider {
        Provider::Anthropic => {
            let key = require_env(
                "ANTHROPIC_API_KEY",
                "export your Anthropic API key to run an agent",
            )?;
            let model = provider.model.as_deref().unwrap_or(DEFAULT_ANTHROPIC_MODEL);
            run_with(anthropic_provider(key, model), prompt, system).await
        }
        Provider::Openai => {
            let key = require_env(
                "OPENAI_API_KEY",
                "export your OpenAI API key to run an agent",
            )?;
            let model = provider
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned());
            run_with(OpenAIProvider::new(key, model), prompt, system).await
        }
        Provider::Gemini => {
            let key = require_gemini_key()?;
            let model = provider
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_GEMINI_MODEL.to_owned());
            run_with(GeminiProvider::new(key, model), prompt, system).await
        }
        Provider::Vertex => run_with(build_vertex(&provider)?, prompt, system).await,
        Provider::Cloudflare => run_with(build_cloudflare(&provider)?, prompt, system).await,
    }
}

async fn chat_async(args: ChatArgs) -> Result<()> {
    let ChatArgs { system, provider } = args;

    match provider.provider {
        Provider::Anthropic => {
            let key = require_env(
                "ANTHROPIC_API_KEY",
                "export your Anthropic API key to run an agent",
            )?;
            let model = provider.model.as_deref().unwrap_or(DEFAULT_ANTHROPIC_MODEL);
            chat_with(anthropic_provider(key, model), system).await
        }
        Provider::Openai => {
            let key = require_env(
                "OPENAI_API_KEY",
                "export your OpenAI API key to run an agent",
            )?;
            let model = provider
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned());
            chat_with(OpenAIProvider::new(key, model), system).await
        }
        Provider::Gemini => {
            let key = require_gemini_key()?;
            let model = provider
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_GEMINI_MODEL.to_owned());
            chat_with(GeminiProvider::new(key, model), system).await
        }
        Provider::Vertex => chat_with(build_vertex(&provider)?, system).await,
        Provider::Cloudflare => chat_with(build_cloudflare(&provider)?, system).await,
    }
}

fn require_gemini_key() -> Result<String> {
    first_env(&["GEMINI_API_KEY", "GOOGLE_API_KEY"]).context(
        "neither GEMINI_API_KEY nor GOOGLE_API_KEY is set; export a Gemini API key to run an agent",
    )
}

/// Run a single prompt against `provider`, streaming the reply to stdout.
async fn run_with<P: LlmProvider + 'static>(
    provider: P,
    prompt: String,
    system: String,
) -> Result<()> {
    let agent = build_agent(provider, system);

    let thread_id = ThreadId::new();
    let _ = agent
        .run(
            thread_id,
            AgentInput::Text(prompt),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await
        .context("agent run failed")?;

    // The streamed deltas above are not newline-terminated.
    println!();
    Ok(())
}

/// Interactive REPL against `provider`, keeping conversation history for the
/// session.
async fn chat_with<P: LlmProvider + 'static>(provider: P, system: String) -> Result<()> {
    let agent = build_agent(provider, system);

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

type CliAgent<P> = agent_sdk::AgentLoop<
    (),
    P,
    agent_sdk::DefaultHooks,
    agent_sdk::InMemoryStore,
    agent_sdk::InMemoryStore,
>;

fn build_agent<P: LlmProvider + 'static>(provider: P, system: String) -> CliAgent<P> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_aliases_resolve_to_expected_models() {
        assert_eq!(
            anthropic_provider("k".to_owned(), "haiku").model(),
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(
            anthropic_provider("k".to_owned(), "sonnet").model(),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            anthropic_provider("k".to_owned(), "opus").model(),
            "claude-opus-4-6"
        );
        assert_eq!(
            anthropic_provider("k".to_owned(), "fable").model(),
            "claude-fable-5"
        );
    }

    #[test]
    fn anthropic_full_model_id_passes_through() {
        assert_eq!(
            anthropic_provider("k".to_owned(), "claude-3-5-haiku-20241022").model(),
            "claude-3-5-haiku-20241022"
        );
    }

    #[test]
    fn resolve_with_env_prefers_explicit_flag() -> Result<()> {
        let got = resolve_with_env(Some("from-flag"), &["UNSET_VAR_X"], "missing")?;
        assert_eq!(got, "from-flag");
        Ok(())
    }

    #[test]
    fn resolve_with_env_ignores_blank_flag_then_errors() {
        // A whitespace-only flag is treated as absent; with no env fallback the
        // resolution surfaces the actionable error rather than panicking.
        let result = resolve_with_env(Some("   "), &["AGENT_SDK_DEFINITELY_UNSET_VAR"], "boom");
        match result {
            Ok(value) => panic!("expected error, got {value:?}"),
            Err(err) => assert!(err.to_string().contains("boom"), "unexpected error: {err}"),
        }
    }

    #[test]
    fn cloudflare_auth_requires_a_credential() {
        match ensure_cf_auth("", None, "ANTHROPIC_API_KEY") {
            Ok(()) => panic!("missing credentials should error"),
            Err(err) => assert!(
                err.to_string().contains("Cloudflare requires credentials"),
                "unexpected error: {err}"
            ),
        }
    }

    #[test]
    fn cloudflare_auth_passes_with_gateway_token() -> Result<()> {
        ensure_cf_auth("", Some("cf-token"), "ANTHROPIC_API_KEY")?;
        Ok(())
    }

    #[test]
    fn cloudflare_auth_passes_with_api_key() -> Result<()> {
        ensure_cf_auth("sk-123", None, "ANTHROPIC_API_KEY")?;
        Ok(())
    }

    #[test]
    fn provider_default_is_anthropic() {
        assert_eq!(Provider::default(), Provider::Anthropic);
    }
}
