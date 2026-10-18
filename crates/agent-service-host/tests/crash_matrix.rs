//! The crash matrix (architecture review §6): each scenario runs a real
//! [`LocalDaemon`] in a child process, kills it with SIGKILL at a named
//! point, boots a fresh daemon on the same database and checks what
//! recovery did.
//!
//! Every scenario runs on SQLite (the `sqlite` feature) and on Postgres
//! (the `postgres` feature, with `TEST_DATABASE_URL` or `DATABASE_URL`
//! set; without a URL the Postgres cases pass without running, as the
//! other Postgres host tests do). Each case gets its own database file or
//! schema.
//!
//! A scenario that fails today is `#[ignore]`d with the failure in its
//! reason and runs as an expected failure: it asserts that the documented
//! failure still reproduces, so the Postgres CI job (`--run-ignored all`)
//! stays green while the bug exists and turns red once it is fixed, and the
//! fix must remove the `#[ignore]` and the expected-failure branch. With
//! `AGENT_SDK_CRASH_MATRIX_KNOWN_FAILURES` set it asserts the real pass
//! condition instead.
//!
//! With the `otel` feature and `AGENT_SDK_CRASH_MATRIX_OBS` set, each
//! case prints the metrics and spans of the recovering process.

#![cfg(any(feature = "sqlite", feature = "postgres"))]

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_sdk_foundation::llm::{
    self, ChatOutcome, ChatRequest, ChatResponse, Content, ContentBlock, StopReason, Usage,
};
use agent_sdk_foundation::{
    AgentEvent, ThreadId, ToolConcurrency, ToolReplay, ToolResult, ToolTier,
};
use agent_sdk_providers::{LlmProvider, StreamBox, StreamDelta};
use agent_server::journal::effect::{EffectKey, EffectRecord, EffectState};
use agent_server::journal::task::{AgentTask, TaskStatus};
use agent_server::worker::ToolEventCollector;
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
use agent_server::worker::{
    CompletionContext, CompletionOutput, CompletionPlan, RunMode, SubagentCompletionHook,
};
use agent_server::{AgentTaskId, ToolTaskBootstrap};
use agent_service_host::config::{EpochRecovery, ServiceConfig, StorageConfig, WorkerConfig};
use agent_service_host::grpc::LocalDaemon;
use agent_service_host::proto::agent::service::v1 as pb;
use agent_service_host::runtime::{
    AllowAllConfirmationPolicy, ExecutionRuntime, StaticProviderResolver, ToolCallExecutor,
};
use agent_service_host::stores::StoreRegistry;
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

type ControlClient = pb::agent_control_service_client::AgentControlServiceClient<Channel>;

const CHILD_ENV: &str = "AGENT_SDK_CRASH_MATRIX_CHILD";
const DIR_ENV: &str = "AGENT_SDK_CRASH_MATRIX_DIR";
const STORAGE_ENV: &str = "AGENT_SDK_CRASH_MATRIX_STORAGE";
#[allow(dead_code)] // Read by an expected-failure branch; none is open today.
const KNOWN_FAILURES_ENV: &str = "AGENT_SDK_CRASH_MATRIX_KNOWN_FAILURES";
/// With the `otel` feature, print the recovery signals each case saw.
#[cfg(feature = "otel")]
const OBS_ENV: &str = "AGENT_SDK_CRASH_MATRIX_OBS";

const UNSAFE_TOOL: &str = "cm_append";
const SAFE_TOOL: &str = "cm_read";
const CONFIRM_TOOL: &str = "cm_deploy";
const QUESTION_TOOL: &str = "ask_user";
const HANG_TOOL: &str = "cm_hang";
const SUBAGENT_TOOL: &str = "cm_subagent";
const SUBAGENT_TASK: &str = "crash-matrix subagent task";
/// Tool children the cancel-tree root starts next to its subagent.
const CANCEL_TOOL_CALLS: usize = 3;
const CALL_ID: &str = "cm-call-1";
const PARTIAL_TEXT: &str = "partial answer that the crash cuts";
const FINAL_TEXT: &str = "final answer";

/// How long a test waits for a condition: several lease periods.
const WAIT: Duration = Duration::from_secs(45);

// ─────────────────────────────────────────────────────────────────────
// Scenarios and phases
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    UnsafeTool,
    SafeTool,
    LlmStream,
    Question,
    Confirmation,
    /// S6 confirmation on a `SingleHost` daemon, which reclaims the dead
    /// process's leases at boot (crash matrix F4).
    ConfirmationSingleHost,
    Fork,
    CancelTree,
    HungTool,
    HungSubagent,
    Overlap,
    MergeBack,
}

impl Scenario {
    const ALL: [Self; 12] = [
        Self::UnsafeTool,
        Self::SafeTool,
        Self::LlmStream,
        Self::Question,
        Self::Confirmation,
        Self::ConfirmationSingleHost,
        Self::Fork,
        Self::CancelTree,
        Self::HungTool,
        Self::HungSubagent,
        Self::Overlap,
        Self::MergeBack,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::UnsafeTool => "s1_unsafe_tool",
            Self::SafeTool => "s2_safe_tool",
            Self::LlmStream => "s3_llm_stream",
            Self::Question => "s6_question",
            Self::Confirmation => "s6_confirmation",
            Self::ConfirmationSingleHost => "s6_confirmation_single_host",
            Self::Fork => "s4_fork",
            Self::CancelTree => "s5_cancel_tree",
            Self::HungTool => "s7_hung_tool",
            Self::HungSubagent => "s7_hung_subagent",
            Self::Overlap => "s8_overlap",
            Self::MergeBack => "s9_merge_back",
        }
    }

    fn parse(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|scenario| scenario.name() == name)
            .with_context(|| format!("unknown crash-matrix scenario {name}"))
    }

    /// The model's first response: the call the crash interrupts.
    fn first_response(self) -> ChatResponse {
        match self {
            Self::UnsafeTool => tool_use(UNSAFE_TOOL, json!({})),
            Self::SafeTool => tool_use(SAFE_TOOL, json!({})),
            Self::Confirmation | Self::ConfirmationSingleHost => tool_use(CONFIRM_TOOL, json!({})),
            Self::Question => tool_use(QUESTION_TOOL, json!({"question": "Which target?"})),
            Self::LlmStream | Self::Fork => text(FINAL_TEXT),
            Self::CancelTree => tool_uses(
                (1..=CANCEL_TOOL_CALLS)
                    .map(|n| (format!("cm-call-{n}"), UNSAFE_TOOL))
                    .chain([("cm-sub-1".to_owned(), SUBAGENT_TOOL)])
                    .collect(),
            ),
            Self::HungTool => tool_use(HANG_TOOL, json!({})),
            Self::HungSubagent | Self::MergeBack => tool_use(SUBAGENT_TOOL, json!({})),
            Self::Overlap => tool_uses(vec![
                ("cm-call-1".to_owned(), UNSAFE_TOOL),
                ("cm-call-2".to_owned(), SAFE_TOOL),
            ]),
        }
    }
}

/// The crashed process blocks at the scenario's crash point; the
/// recovering process runs straight through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Crash,
    Recover,
}

const fn usage() -> Usage {
    Usage {
        served_speed: None,
        input_tokens: 1,
        output_tokens: 1,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
        request_input_tokens: None,
    }
}

fn tool_use(name: &str, input: Value) -> ChatResponse {
    ChatResponse {
        id: format!("cm-response-{name}"),
        content: vec![ContentBlock::ToolUse {
            id: CALL_ID.into(),
            name: name.into(),
            input,
            thought_signature: None,
        }],
        model: "cm-model".into(),
        stop_reason: Some(StopReason::ToolUse),
        usage: usage(),
    }
}

fn tool_uses(calls: Vec<(String, &str)>) -> ChatResponse {
    ChatResponse {
        id: "cm-response-batch".into(),
        content: calls
            .into_iter()
            .map(|(id, name)| ContentBlock::ToolUse {
                id,
                name: name.into(),
                input: json!({}),
                thought_signature: None,
            })
            .collect(),
        model: "cm-model".into(),
        stop_reason: Some(StopReason::ToolUse),
        usage: usage(),
    }
}

/// Whether `request` is a subagent child's turn: its task is in the
/// transcript.
fn is_subagent_turn(request: &ChatRequest) -> bool {
    request
        .messages
        .iter()
        .any(|message| {
            match &message.content {
        Content::Text(text) => text.contains(SUBAGENT_TASK),
        Content::Blocks(blocks) => blocks.iter().any(|block| {
            matches!(block, ContentBlock::Text { text } if text.contains(SUBAGENT_TASK))
        }),
    }
        })
}

fn text(text: &str) -> ChatResponse {
    ChatResponse {
        id: "cm-response-text".into(),
        content: vec![ContentBlock::Text { text: text.into() }],
        model: "cm-model".into(),
        stop_reason: Some(StopReason::EndTurn),
        usage: usage(),
    }
}

fn tool_results(request: &ChatRequest) -> Vec<(String, bool)> {
    request
        .messages
        .iter()
        .filter_map(|message| match &message.content {
            Content::Blocks(blocks) => Some(blocks),
            Content::Text(_) => None,
        })
        .flatten()
        .filter_map(|block| match block {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => Some((content.clone(), is_error.unwrap_or(false))),
            _ => None,
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// The scripted model and tools
// ─────────────────────────────────────────────────────────────────────

/// Marker files in the case directory: `ready` says the crash point is
/// reached, `runs` gets one line each time a tool body starts.
struct CaseDir(PathBuf);

impl CaseDir {
    fn ready(&self) -> PathBuf {
        self.0.join("ready")
    }

    fn runs(&self) -> PathBuf {
        self.0.join("runs")
    }

    fn endpoint(&self) -> PathBuf {
        self.0.join("endpoint")
    }

    fn record_run(&self, tool: &str) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.runs())?;
        writeln!(file, "{tool}")?;
        file.sync_all()?;
        Ok(())
    }

    fn run_lines(&self) -> Result<Vec<String>> {
        match std::fs::read_to_string(self.runs()) {
            Ok(text) => Ok(text.lines().map(str::to_owned).collect()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(error) => Err(error.into()),
        }
    }

    fn signal_ready(&self, what: &str) -> Result<()> {
        std::fs::write(self.ready(), what)?;
        Ok(())
    }

    /// Set once a subagent child's model call is under way.
    fn subagent_ready(&self) -> PathBuf {
        self.0.join("ready-subagent")
    }
}

struct MatrixProvider {
    scenario: Scenario,
    phase: Phase,
    dir: Arc<CaseDir>,
    /// The tool results the model was sent, one entry per request.
    seen: Mutex<Vec<Vec<(String, bool)>>>,
}

impl MatrixProvider {
    /// The final text once the model has a tool result, else the
    /// scenario's first response.
    fn respond(&self, request: &ChatRequest) -> ChatResponse {
        if is_subagent_turn(request) {
            return text(FINAL_TEXT);
        }
        let results = tool_results(request);
        let answered = !results.is_empty();
        if let Ok(mut seen) = self.seen.lock() {
            seen.push(results);
        }
        if answered {
            text(FINAL_TEXT)
        } else {
            self.scenario.first_response()
        }
    }

    fn requests(&self) -> Vec<Vec<(String, bool)>> {
        self.seen
            .lock()
            .map(|seen| seen.clone())
            .unwrap_or_default()
    }
}

#[async_trait]
impl LlmProvider for MatrixProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        Ok(ChatOutcome::Success(self.respond(&request)))
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        let hung_subagent = is_subagent_turn(&request)
            && (self.scenario == Scenario::HungSubagent
                || (self.scenario == Scenario::CancelTree && self.phase == Phase::Crash));
        if hung_subagent {
            // A subagent child whose model never answers; its lease keeps
            // beating.
            return Box::pin(async_stream::stream! {
                if let Err(error) = std::fs::write(self.dir.subagent_ready(), "subagent") {
                    yield Err(error.into());
                    return;
                }
                tokio::time::sleep(Duration::from_secs(600)).await;
                yield Err(anyhow::anyhow!("the subagent stream was not killed"));
            });
        }
        if self.scenario != Scenario::LlmStream || self.phase != Phase::Crash {
            return response_stream(self.respond(&request));
        }
        Box::pin(async_stream::stream! {
            yield Ok(StreamDelta::TextDelta {
                delta: PARTIAL_TEXT.into(),
                block_index: 0,
            });
            // The worker has taken the first delta; the crash lands mid-stream.
            if let Err(error) = self.dir.signal_ready("llm stream") {
                yield Err(error);
                return;
            }
            tokio::time::sleep(Duration::from_secs(600)).await;
            yield Err(anyhow::anyhow!("the stream was not killed"));
        })
    }

    fn model(&self) -> &'static str {
        "cm-model"
    }

    fn provider(&self) -> &'static str {
        "cm"
    }
}

/// S9's completion hook: one `merge_back` git step. The crashed process
/// publishes the invocation id from inside the hook (after its `Started`
/// record committed) and waits to be killed; the recovering process records
/// the attempt, mode and prior record it ran with, and finishes the step.
struct MatrixHook {
    phase: Phase,
    dir: Arc<CaseDir>,
}

#[async_trait]
impl SubagentCompletionHook for MatrixHook {
    fn plan(&self, _ctx: &CompletionContext) -> Option<CompletionPlan> {
        Some(CompletionPlan {
            step: "merge_back".into(),
            repo: "repo_1".into(),
            unit: "unit_1".into(),
        })
    }

    async fn run(
        &self,
        ctx: &CompletionContext,
        prior: Option<&EffectRecord>,
        mode: RunMode,
    ) -> Result<CompletionOutput> {
        if self.phase == Phase::Crash {
            self.dir.signal_ready(ctx.invocation.as_str())?;
            tokio::time::sleep(Duration::from_secs(600)).await;
            bail!("the crash-matrix child was not killed while its hook ran");
        }
        self.dir.record_run(&format!(
            "merge_back attempt={} mode={mode:?} prior={}",
            ctx.attempt,
            prior.map_or("none", |record| record.state.outcome_label()),
        ))?;
        Ok(CompletionOutput {
            result: ToolResult::success("merged unit_1"),
            state: EffectState::Completed,
            extra_writes: Vec::new(),
            park: None,
        })
    }
}

/// Stream a scripted response as a provider does: its blocks, then the
/// usage and the stop reason.
fn response_stream(response: ChatResponse) -> StreamBox<'static> {
    Box::pin(async_stream::stream! {
        for (block_index, block) in response.content.into_iter().enumerate() {
            match block {
                ContentBlock::Text { text } => {
                    yield Ok(StreamDelta::TextDelta { delta: text, block_index });
                }
                ContentBlock::ToolUse { id, name, input, thought_signature } => {
                    yield Ok(StreamDelta::ToolUseStart {
                        id: id.clone(),
                        name,
                        block_index,
                        thought_signature,
                    });
                    yield Ok(StreamDelta::ToolInputDelta {
                        id,
                        delta: input.to_string(),
                        block_index,
                    });
                }
                _ => {}
            }
        }
        yield Ok(StreamDelta::Usage(response.usage));
        yield Ok(StreamDelta::Done {
            stop_reason: response.stop_reason,
            served_route: None,
            served_model: None,
        });
    })
}

struct MatrixTools {
    phase: Phase,
    dir: Arc<CaseDir>,
}

#[async_trait]
impl ToolCallExecutor for MatrixTools {
    async fn execute_tool_call(
        &self,
        bootstrap: &ToolTaskBootstrap,
        _collector: ToolEventCollector,
        cancel: CancellationToken,
        _artifact_store: Option<Arc<agent_sdk::ArtifactStore>>,
    ) -> Result<ToolResult> {
        let name = bootstrap.tool_call.name.as_str();
        // The protected side effect happens before the crash point.
        self.dir.record_run(name)?;
        if name == HANG_TOOL || (self.phase == Phase::Crash && name != CONFIRM_TOOL) {
            self.dir.signal_ready(bootstrap.task_id.as_str())?;
            tokio::select! {
                () = cancel.cancelled() => return Ok(ToolResult::error(format!("{name} cancelled"))),
                () = tokio::time::sleep(Duration::from_secs(600)) => {}
            }
            bail!("the tool was not killed");
        }
        Ok(ToolResult::success(format!("{name} ran")))
    }
}

fn advertised_tools() -> Vec<llm::Tool> {
    let tool = |name: &str, tier, concurrency, replay| llm::Tool {
        name: name.into(),
        description: format!("crash-matrix tool {name}"),
        input_schema: json!({"type": "object"}),
        display_name: name.into(),
        tier,
        concurrency,
        replay,
    };
    vec![
        // Read-only concurrency so a batch runs its calls side by side;
        // replay is what the matrix measures.
        tool(
            UNSAFE_TOOL,
            ToolTier::Observe,
            ToolConcurrency::ReadOnly,
            ToolReplay::Unsafe,
        ),
        tool(
            HANG_TOOL,
            ToolTier::Observe,
            ToolConcurrency::ReadOnly,
            ToolReplay::Unsafe,
        ),
        // The SDK refuses a subagent spawn that is not confirm-tier.
        tool(
            SUBAGENT_TOOL,
            ToolTier::Confirm,
            ToolConcurrency::ReadOnly,
            ToolReplay::Unsafe,
        ),
        tool(
            SAFE_TOOL,
            ToolTier::Observe,
            ToolConcurrency::ReadOnly,
            ToolReplay::Safe,
        ),
        tool(
            CONFIRM_TOOL,
            ToolTier::Confirm,
            ToolConcurrency::Exclusive,
            ToolReplay::Unsafe,
        ),
        tool(
            QUESTION_TOOL,
            ToolTier::Observe,
            ToolConcurrency::Exclusive,
            ToolReplay::Safe,
        ),
    ]
}

/// Routes the subagent tool into a durable subagent invocation with a
/// child thread named after its parent, as the host tests'
/// `DeadlineSpawnSelector` does. Its `timeout_ms` is a silence budget, not
/// an age limit: S7 sets it to [`STALL_BUDGET`].
struct MatrixSelector;

#[async_trait]
impl agent_server::worker::SubagentSpawnSelector for MatrixSelector {
    async fn decide(
        &self,
        parent_thread_id: &ThreadId,
        tool_calls: &[agent_sdk_foundation::PendingToolCallInfo],
    ) -> Result<Vec<agent_server::worker::SubagentSpawnDecision>> {
        use agent_server::worker::subagent_spawn_selector::SubagentSpawnPlan;
        use agent_server::worker::{
            SubagentCapabilityRequest, SubagentSpawnDecision, SubagentSpawnRequest,
        };
        Ok(tool_calls
            .iter()
            .map(|call| {
                if call.name != SUBAGENT_TOOL {
                    return SubagentSpawnDecision::SpawnAsTool;
                }
                SubagentSpawnDecision::SpawnAsSubagent {
                    plan: Box::new(SubagentSpawnPlan {
                        request: SubagentSpawnRequest::new(
                            SUBAGENT_TASK,
                            SubagentCapabilityRequest::new("research"),
                        ),
                        spec: subagent_spec(),
                        child_thread_id: ThreadId::from_string(format!("{parent_thread_id}-sub")),
                        child_root_input: Vec::new(),
                        child_caller_metadata: None,
                    }),
                }
            })
            .collect())
    }
}

fn subagent_spec() -> agent_server::worker::EffectiveSubagentSpec {
    use agent_server::worker::{
        EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
        InheritedSubagentPolicy, SubagentCapabilityProfile, SubagentSandboxPolicy,
    };
    use std::collections::{BTreeMap, BTreeSet};

    const TIMEOUT_MS: u64 = STALL_BUDGET.as_secs() * 1_000;
    let capabilities = BTreeSet::from(["read_file".to_owned()]);
    EffectiveSubagentSpec {
        task: SUBAGENT_TASK.to_owned(),
        prompt: String::new(),
        model: "cm-model".into(),
        max_turns: 5,
        timeout_ms: TIMEOUT_MS,
        depth: 1,
        max_parallel_subagents: 1,
        nickname: None,
        sandbox: SubagentSandboxPolicy::read_only(),
        mcp: EffectiveSubagentMcpPolicy::default(),
        audit_provenance: None,
        inherited_policy: InheritedSubagentPolicy {
            default_model: "cm-model".into(),
            allowed_models: BTreeSet::from(["cm-model".to_owned()]),
            default_max_turns: 5,
            max_turns: 5,
            default_timeout_ms: TIMEOUT_MS,
            max_timeout_ms: TIMEOUT_MS,
            capability_profiles: BTreeMap::from([(
                "research".to_owned(),
                SubagentCapabilityProfile {
                    capabilities: capabilities.clone(),
                    sandbox: SubagentSandboxPolicy::read_only(),
                    allowed_mcp_servers: BTreeSet::new(),
                },
            )]),
            allowed_capabilities: capabilities.clone(),
            max_depth: 3,
            max_parallel_subagents: 1,
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: BTreeSet::new(),
            audit_provider: "cm".into(),
        },
        capabilities: EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: capabilities,
        },
    }
}

fn definition() -> AgentDefinition {
    AgentDefinition {
        provider: "cm".into(),
        model: "cm-model".into(),
        system_prompt: "Crash matrix.".into(),
        max_tokens: 256,
        tools: advertised_tools(),
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Backends
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Backend {
    #[cfg(feature = "sqlite")]
    Sqlite,
    #[cfg(feature = "postgres")]
    Postgres,
}

impl Backend {
    fn from_name(name: &str) -> Self {
        match name {
            #[cfg(feature = "sqlite")]
            "sqlite" => Self::Sqlite,
            #[cfg(feature = "postgres")]
            _ => Self::Postgres,
            #[cfg(not(feature = "postgres"))]
            _ => Self::Sqlite,
        }
    }
}

/// A database that outlives the processes of one case.
struct CaseDb {
    storage: StorageConfig,
    #[cfg(feature = "postgres")]
    _schema: Option<PgSchema>,
}

#[cfg(feature = "postgres")]
struct PgSchema {
    database_url: String,
    schema: String,
}

#[cfg(feature = "postgres")]
impl Drop for PgSchema {
    fn drop(&mut self) {
        let url = self.database_url.clone();
        let schema = self.schema.clone();
        let _ = std::thread::spawn(move || {
            use sqlx::Connection;
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            runtime.block_on(async move {
                let Ok(mut conn) = sqlx::PgConnection::connect(&url).await else {
                    return;
                };
                let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                    "DROP SCHEMA IF EXISTS {schema} CASCADE"
                )))
                .execute(&mut conn)
                .await;
            });
        })
        .join();
    }
}

impl CaseDb {
    /// A fresh database for `backend`, or `None` when the backend has no
    /// test database configured.
    async fn create(backend: Backend, dir: &Path) -> Result<Option<Self>> {
        match backend {
            #[cfg(feature = "sqlite")]
            Backend::Sqlite => Ok(Some(Self {
                storage: StorageConfig {
                    backend: agent_service_host::config::StorageBackend::Sqlite {
                        path: Some(dir.join("journal.db").display().to_string()),
                        max_connections: None,
                    },
                    ..StorageConfig::default()
                },
                #[cfg(feature = "postgres")]
                _schema: None,
            })),
            #[cfg(feature = "postgres")]
            Backend::Postgres => {
                use sqlx::Connection;
                // Postgres keeps its state in a schema, not the case directory.
                let _ = dir;
                let Ok(database_url) =
                    std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
                else {
                    return Ok(None);
                };
                let schema = format!("crash_matrix_{}", uuid::Uuid::new_v4().simple());
                let mut admin = sqlx::PgConnection::connect(&database_url)
                    .await
                    .context("connect postgres admin for the crash matrix")?;
                sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
                    .execute(&mut admin)
                    .await
                    .with_context(|| format!("create crash-matrix schema {schema}"))?;
                Ok(Some(Self {
                    storage: StorageConfig {
                        backend: agent_service_host::config::StorageBackend::Postgres,
                        postgres: agent_service_host::config::PostgresStorageConfig {
                            database_url: Some(database_url.clone()),
                            schema: Some(schema.clone()),
                            max_connections: 8,
                        },
                    },
                    _schema: Some(PgSchema {
                        database_url,
                        schema,
                    }),
                }))
            }
        }
    }
}

/// Short leases so a dead process's work is reclaimed within seconds.
/// `SingleHost` scenarios also reclaim the dead process's leases at boot.
fn service_config(storage: StorageConfig, scenario: Scenario) -> ServiceConfig {
    ServiceConfig {
        storage,
        worker: WorkerConfig {
            pool_size: 4,
            lease_duration_secs: 3,
            heartbeat_interval_secs: 1,
            sweep_interval_secs: 1,
            acquisition_interval_secs: 1,
            tool_stall_budget_secs: STALL_BUDGET.as_secs(),
            epoch_recovery: if scenario == Scenario::ConfirmationSingleHost {
                EpochRecovery::SingleHost
            } else {
                EpochRecovery::default()
            },
            ..WorkerConfig::default()
        },
        ..ServiceConfig::default()
    }
}

struct Daemon {
    host: LocalDaemon,
    provider: Arc<MatrixProvider>,
    signals: Option<signals::Capture>,
    backend: &'static str,
}

impl Daemon {
    /// Print what the recovering process emitted, when asked to.
    fn report(&self, case: &str) {
        if let Some(signals) = &self.signals {
            signals.report(&format!("{} {case}", self.backend));
        }
    }
}

/// An in-memory meter and tracer for the recovering process, so a case
/// can say which spans and metrics show what recovery did.
mod signals {
    #[cfg(feature = "otel")]
    pub struct Capture {
        meters: opentelemetry_sdk::metrics::SdkMeterProvider,
        metrics: opentelemetry_sdk::metrics::InMemoryMetricExporter,
        spans: opentelemetry_sdk::trace::InMemorySpanExporter,
    }

    #[cfg(not(feature = "otel"))]
    pub struct Capture;

    #[cfg(feature = "otel")]
    pub fn install() -> Option<Capture> {
        use opentelemetry_sdk::metrics::{
            InMemoryMetricExporter, PeriodicReader, SdkMeterProvider,
        };
        use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
        std::env::var_os(super::OBS_ENV)?;
        let spans = InMemorySpanExporter::default();
        let tracer = SdkTracerProvider::builder()
            .with_simple_exporter(spans.clone())
            .build();
        opentelemetry::global::set_tracer_provider(tracer);
        let metrics = InMemoryMetricExporter::default();
        let meters = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(metrics.clone()).build())
            .build();
        opentelemetry::global::set_meter_provider(meters.clone());
        agent_server::observability::ServerMetrics::reset_for_testing();
        agent_service_host::observability::HostMetrics::reset_for_testing();
        Some(Capture {
            meters,
            metrics,
            spans,
        })
    }

    #[cfg(not(feature = "otel"))]
    pub fn install() -> Option<Capture> {
        None
    }

    impl Capture {
        /// Print every metric point and span, one line each.
        #[cfg(feature = "otel")]
        pub fn report(&self, case: &str) {
            use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
            let _ = self.meters.force_flush();
            let mut lines = std::collections::BTreeSet::new();
            // Each export is a cumulative snapshot; the last one has the totals.
            let exports = self.metrics.get_finished_metrics().unwrap_or_default();
            for resource in exports.iter().rev().take(1) {
                for scope in resource.scope_metrics() {
                    for metric in scope.metrics() {
                        let points: Vec<String> = match metric.data() {
                            AggregatedMetrics::U64(MetricData::Sum(sum)) => sum
                                .data_points()
                                .map(|point| {
                                    format!("{}={}", attrs(point.attributes()), point.value())
                                })
                                .collect(),
                            AggregatedMetrics::I64(MetricData::Sum(sum)) => sum
                                .data_points()
                                .map(|point| {
                                    format!("{}={}", attrs(point.attributes()), point.value())
                                })
                                .collect(),
                            AggregatedMetrics::F64(MetricData::Histogram(histogram)) => histogram
                                .data_points()
                                .map(|point| {
                                    format!("{}=n{}", attrs(point.attributes()), point.count())
                                })
                                .collect(),
                            _ => vec!["(gauge)".to_owned()],
                        };
                        for point in points {
                            lines.insert(format!("metric {} {point}", metric.name()));
                        }
                    }
                }
            }
            for span in self.spans.get_finished_spans().unwrap_or_default() {
                let events: Vec<String> = span
                    .events
                    .iter()
                    .map(|event| event.name.to_string())
                    .collect();
                lines.insert(format!(
                    "span {} {} events={events:?}",
                    span.name,
                    attrs(span.attributes.iter())
                ));
            }
            for line in lines {
                println!("signals {case}: {line}");
            }
        }

        #[cfg(not(feature = "otel"))]
        pub const fn report(&self, _case: &str) {}
    }

    /// Attributes without ids, so equal points collapse into one line.
    #[cfg(feature = "otel")]
    fn attrs<'a>(iter: impl Iterator<Item = &'a opentelemetry::KeyValue>) -> String {
        let pairs: Vec<String> = iter
            .filter(|kv| kv.key.as_str().rsplit(['.', '_']).next() != Some("id"))
            .map(|kv| format!("{}={}", kv.key.as_str(), kv.value))
            .collect();
        format!("{{{}}}", pairs.join(","))
    }
}

async fn start_daemon(
    storage: StorageConfig,
    scenario: Scenario,
    phase: Phase,
    dir: Arc<CaseDir>,
) -> Result<Daemon> {
    let signals = if phase == Phase::Recover {
        signals::install()
    } else {
        None
    };
    let provider = Arc::new(MatrixProvider {
        scenario,
        phase,
        dir: Arc::clone(&dir),
        seen: Mutex::new(Vec::new()),
    });
    let resolver = Arc::new(StaticProviderResolver::new());
    resolver.set_fallback(Arc::clone(&provider) as Arc<dyn LlmProvider>)?;
    let runtime = Arc::new(
        ExecutionRuntime::new(
            resolver,
            Arc::new(MatrixTools {
                phase,
                dir: Arc::clone(&dir),
            }),
            Arc::new(AllowAllConfirmationPolicy),
        )
        .with_subagent_spawn_selector(Arc::new(MatrixSelector)),
    );
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(definition()));
    let config_backend = storage.backend.clone();
    let hook = (scenario == Scenario::MergeBack)
        .then(|| Arc::new(MatrixHook { phase, dir }) as Arc<dyn SubagentCompletionHook>);
    let daemon = LocalDaemon::start_with_store_decorator(
        service_config(storage, scenario),
        registry,
        runtime,
        move |stores| match hook {
            Some(hook) => stores.with_subagent_completion_hook(hook),
            None => stores,
        },
    )
    .await?;
    Ok(Daemon {
        host: daemon,
        provider,
        signals,
        backend: storage_name(&config_backend),
    })
}

const fn storage_name(backend: &agent_service_host::config::StorageBackend) -> &'static str {
    match backend {
        agent_service_host::config::StorageBackend::Postgres => "postgres",
        agent_service_host::config::StorageBackend::InMemory => "memory",
        agent_service_host::config::StorageBackend::Sqlite { .. } => "sqlite",
    }
}

async fn inspection(storage: &StorageConfig) -> Result<StoreRegistry> {
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(definition()));
    let stores = StoreRegistry::from_config(storage, registry)?;
    stores.initialize().await?;
    Ok(stores)
}

// ─────────────────────────────────────────────────────────────────────
// The crashed process
// ─────────────────────────────────────────────────────────────────────

/// Child-process half: run a daemon until the parent kills it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crash_matrix_child() -> Result<()> {
    let Some(scenario) = std::env::var_os(CHILD_ENV) else {
        return Ok(());
    };
    let scenario = Scenario::parse(&scenario.to_string_lossy())?;
    let dir = Arc::new(CaseDir(PathBuf::from(
        std::env::var(DIR_ENV).context("missing case directory")?,
    )));
    let storage: StorageConfig =
        serde_json::from_str(&std::env::var(STORAGE_ENV).context("missing storage config")?)?;
    #[cfg(feature = "failpoints")]
    let _failpoints = fail::FailScenario::setup();
    #[cfg(feature = "failpoints")]
    if scenario == Scenario::Fork {
        fail::cfg("fork.after_commit", "pause").map_err(anyhow::Error::msg)?;
    }
    let daemon = start_daemon(storage, scenario, Phase::Crash, Arc::clone(&dir)).await?;
    let endpoint = dir.endpoint();
    std::fs::write(endpoint.with_extension("tmp"), daemon.host.endpoint())?;
    std::fs::rename(endpoint.with_extension("tmp"), endpoint)?;
    tokio::time::sleep(Duration::from_secs(600)).await;
    bail!("the crash-matrix child was not killed")
}

struct CrashedHost {
    child: Child,
}

impl CrashedHost {
    fn spawn(scenario: Scenario, dir: &CaseDir, storage: &StorageConfig) -> Result<Self> {
        let child = Command::new(std::env::current_exe()?)
            .arg("crash_matrix_child")
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, scenario.name())
            .env(DIR_ENV, &dir.0)
            .env(STORAGE_ENV, serde_json::to_string(storage)?)
            .spawn()
            .context("spawn crash-matrix child")?;
        Ok(Self { child })
    }

    async fn endpoint(&mut self, dir: &CaseDir) -> Result<String> {
        wait_file(&mut self.child, &dir.endpoint()).await
    }

    fn kill_nine(mut self) -> Result<()> {
        let status = Command::new("kill")
            .arg("-9")
            .arg(self.child.id().to_string())
            .status()
            .context("send kill -9")?;
        ensure!(status.success(), "kill -9 failed with {status}");
        let _ = self.child.wait()?;
        Ok(())
    }
}

impl Drop for CrashedHost {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn wait_file(child: &mut Child, path: &Path) -> Result<String> {
    let deadline = tokio::time::Instant::now() + WAIT;
    loop {
        if let Ok(text) = std::fs::read_to_string(path)
            && !text.is_empty()
        {
            return Ok(text);
        }
        if let Some(status) = child.try_wait()? {
            bail!(
                "the crash-matrix child exited with {status} before {}",
                path.display()
            );
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Client helpers
// ─────────────────────────────────────────────────────────────────────

async fn connect(endpoint: &str) -> Result<ControlClient> {
    let channel = Channel::from_shared(endpoint.to_owned())?
        .connect()
        .await
        .context("connect to the daemon")?;
    Ok(ControlClient::new(channel))
}

async fn start_turn(control: &mut ControlClient) -> Result<(String, String)> {
    let thread = control
        .create_thread(pb::CreateThreadRequest {
            request_id: "cm-create".into(),
            thread_id: String::new(),
        })
        .await?
        .into_inner()
        .thread
        .and_then(|view| view.thread)
        .context("create_thread returned no thread")?
        .thread_id;
    let task = control
        .submit_thread_work(pb::SubmitThreadWorkRequest {
            request_id: "cm-submit".into(),
            thread_id: thread.clone(),
            input: vec![pb::UserInputItem {
                item: Some(pb::user_input_item::Item::Text("run the scenario".into())),
            }],
            caller_metadata: String::new(),
            disposition: pb::SubmitDisposition::NextTurn as i32,
        })
        .await?
        .into_inner()
        .task
        .context("submit_thread_work returned no task")?;
    Ok((thread, task.task_id))
}

async fn wait_for_task(
    stores: &StoreRegistry,
    task: &AgentTaskId,
    what: &str,
    done: impl Fn(&AgentTask) -> bool,
) -> Result<AgentTask> {
    wait_for_task_within(stores, task, what, WAIT, done).await
}

async fn wait_for_task_within(
    stores: &StoreRegistry,
    task: &AgentTaskId,
    what: &str,
    wait: Duration,
    done: impl Fn(&AgentTask) -> bool,
) -> Result<AgentTask> {
    let deadline = tokio::time::Instant::now() + wait;
    loop {
        if let Some(row) = stores.task_store.get(task).await?
            && done(&row)
        {
            return Ok(row);
        }
        if tokio::time::Instant::now() >= deadline {
            let row = stores.task_store.get(task).await?;
            let rows = all_rows(stores).await?;
            let rows: Vec<String> = rows
                .iter()
                .map(|row| {
                    format!(
                        "{} {:?} {:?} attempt {} parent {:?} error {:?}",
                        row.id, row.kind, row.status, row.attempt, row.parent_id, row.last_error
                    )
                })
                .collect();
            bail!(
                "timed out waiting for task {task} to be {what}: {row:?}\nall rows:\n{}",
                rows.join("\n")
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn tool_call_ends(events: &[AgentEvent]) -> Vec<ToolResult> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallEnd { id, result, .. } if id == CALL_ID => Some(result.clone()),
            _ => None,
        })
        .collect()
}

async fn thread_events(stores: &StoreRegistry, thread: &ThreadId) -> Result<Vec<AgentEvent>> {
    Ok(stores
        .event_repo
        .get_events(thread)
        .await?
        .into_iter()
        .map(|event| event.event)
        .collect())
}

/// The rows of `thread` that are not terminal.
async fn live_rows(stores: &StoreRegistry, thread: &ThreadId) -> Result<Vec<AgentTask>> {
    Ok(stores
        .task_store
        .list_by_thread(thread)
        .await?
        .into_iter()
        .filter(|task| !task.status.is_terminal())
        .collect())
}

fn assistant_texts(messages: &[llm::Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|message| message.role == llm::Role::Assistant)
        .flat_map(|message| match &message.content {
            Content::Text(text) => vec![text.clone()],
            Content::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect(),
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// One case: crash, recover, observe
// ─────────────────────────────────────────────────────────────────────

/// What a case saw after recovery.
struct Recovered {
    stores: StoreRegistry,
    thread: ThreadId,
    root: AgentTask,
    tool_child: Option<AgentTaskId>,
    runs: Vec<String>,
    requests: Vec<Vec<(String, bool)>>,
    _daemon: Daemon,
    _db: CaseDb,
    _temp: tempfile::TempDir,
}

/// Crash the scenario's first daemon at its crash point, boot a second
/// one on the same database and run the turn to an end.
async fn crash_and_recover(backend: Backend, scenario: Scenario) -> Result<Option<Recovered>> {
    let temp = tempfile::tempdir()?;
    let dir = Arc::new(CaseDir(temp.path().to_owned()));
    let Some(db) = CaseDb::create(backend, temp.path()).await? else {
        return Ok(None);
    };

    let mut crashed = CrashedHost::spawn(scenario, &dir, &db.storage)?;
    let endpoint = crashed.endpoint(&dir).await?;
    let mut control = connect(&endpoint).await?;
    let (thread, root) = start_turn(&mut control).await?;
    let thread = ThreadId::from_string(&thread);
    let root = AgentTaskId::from_string(&root);

    let stores = inspection(&db.storage).await?;
    let tool_child = match scenario {
        Scenario::Question => {
            wait_for_task(&stores, &root, "awaiting its question", |task| {
                task.status == TaskStatus::AwaitingQuestion
            })
            .await?;
            None
        }
        Scenario::Confirmation | Scenario::ConfirmationSingleHost => {
            let child = wait_for_child(&stores, &root, TaskStatus::AwaitingConfirmation).await?;
            Some(child)
        }
        Scenario::UnsafeTool
        | Scenario::SafeTool
        | Scenario::LlmStream
        | Scenario::Fork
        | Scenario::CancelTree
        | Scenario::HungTool
        | Scenario::HungSubagent
        | Scenario::Overlap
        | Scenario::MergeBack => {
            let ready = wait_file(&mut crashed.child, &dir.ready()).await?;
            (scenario != Scenario::LlmStream).then(|| AgentTaskId::from_string(ready.trim()))
        }
    };
    drop(control);
    crashed.kill_nine()?;

    let daemon = start_daemon(
        db.storage.clone(),
        scenario,
        Phase::Recover,
        Arc::clone(&dir),
    )
    .await?;
    answer_prompt(
        &daemon,
        &stores,
        scenario,
        &thread,
        &root,
        tool_child.as_ref(),
    )
    .await?;

    let root_row =
        wait_for_task(&stores, &root, "terminal", |task| task.status.is_terminal()).await?;
    // Let a duplicate run, if recovery starts one, reach its marker.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let runs = dir.run_lines()?;
    let requests = daemon.provider.requests();
    daemon.report(scenario.name());
    Ok(Some(Recovered {
        stores,
        thread,
        root: root_row,
        tool_child,
        runs,
        requests,
        _daemon: daemon,
        _db: db,
        _temp: temp,
    }))
}

/// After the restart, answer the prompt a S6 case parked on: the prompt
/// must be visible, and a retried answer is not a second answer.
async fn answer_prompt(
    daemon: &Daemon,
    stores: &StoreRegistry,
    scenario: Scenario,
    thread: &ThreadId,
    root: &AgentTaskId,
    tool_child: Option<&AgentTaskId>,
) -> Result<()> {
    let mut control = connect(&daemon.host.endpoint()).await?;
    match scenario {
        Scenario::Question => {
            let parked = wait_for_task(stores, root, "awaiting its question", |task| {
                task.status == TaskStatus::AwaitingQuestion
            })
            .await
            .context("the question is not visible after the restart")?;
            let answer = pb::AnswerQuestionRequest {
                request_id: "cm-answer".into(),
                thread_id: thread.to_string(),
                task_id: parked.id.as_str().to_owned(),
                answers: vec![pb::QuestionAnswerEntry {
                    tool_call_id: CALL_ID.into(),
                    answer: "prod".into(),
                }],
            };
            control.answer_question(answer.clone()).await?;
            // The same request again is a retry, not a second answer.
            let _ = control.answer_question(answer).await;
        }
        Scenario::Confirmation | Scenario::ConfirmationSingleHost => {
            let child = tool_child.context("no confirmation child")?;
            wait_for_task(stores, child, "awaiting confirmation", |task| {
                task.status == TaskStatus::AwaitingConfirmation
            })
            .await
            .context("the confirmation is not visible after the restart")?;
            let decide = pb::DecideConfirmationRequest {
                request_id: "cm-decide".into(),
                thread_id: thread.to_string(),
                task_id: child.as_str().to_owned(),
                decision: Some(pb::ConfirmationDecision {
                    decision: Some(pb::confirmation_decision::Decision::Approved(
                        pb::ApprovedConfirmation::default(),
                    )),
                }),
                automatic: false,
                prompt_id: String::new(),
            };
            control.decide_confirmation(decide.clone()).await?;
            let _ = control.decide_confirmation(decide).await;
        }
        Scenario::UnsafeTool
        | Scenario::SafeTool
        | Scenario::LlmStream
        | Scenario::Fork
        | Scenario::CancelTree
        | Scenario::HungTool
        | Scenario::HungSubagent
        | Scenario::Overlap
        | Scenario::MergeBack => {}
    }
    Ok(())
}

async fn wait_for_child(
    stores: &StoreRegistry,
    root: &AgentTaskId,
    status: TaskStatus,
) -> Result<AgentTaskId> {
    let deadline = tokio::time::Instant::now() + WAIT;
    loop {
        if let Some(child) = stores
            .task_store
            .list_children(root)
            .await?
            .into_iter()
            .find(|child| child.status == status)
        {
            return Ok(child.id);
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for a {status:?} child of {root}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Pass conditions (architecture review §6.1)
// ─────────────────────────────────────────────────────────────────────

/// S1: the marker has one line, the model is told the call was
/// interrupted, clients see the call end once, and the turn ends.
async fn s1_unsafe_tool(backend: Backend) -> Result<()> {
    let Some(case) = crash_and_recover(backend, Scenario::UnsafeTool).await? else {
        return Ok(());
    };
    ensure!(
        case.runs == [UNSAFE_TOOL],
        "the unsafe tool ran {} times: {:?}",
        case.runs.len(),
        case.runs
    );
    let child = case.tool_child.as_ref().context("no tool child")?;
    let child = case
        .stores
        .task_store
        .get(child)
        .await?
        .context("tool child lost")?;
    ensure!(
        child.status == TaskStatus::Failed,
        "the interrupted tool child is {:?}",
        child.status
    );
    let ends = tool_call_ends(&thread_events(&case.stores, &case.thread).await?);
    let [end] = ends.as_slice() else {
        bail!("expected one ToolCallEnd, got {ends:?}");
    };
    ensure!(
        !end.success,
        "the interrupted call ended as a success: {end:?}"
    );
    ensure!(
        case.requests
            .iter()
            .flatten()
            .any(|(content, is_error)| *is_error && content.to_lowercase().contains("interrupted")),
        "the model was not told the call was interrupted: {:?}",
        case.requests
    );
    ensure!(
        case.root.status == TaskStatus::Completed,
        "the root turn is {:?}: {:?}",
        case.root.status,
        case.root.last_error
    );
    let live = live_rows(&case.stores, &case.thread).await?;
    ensure!(live.is_empty(), "rows left live: {live:?}");
    Ok(())
}

/// S2: the safe tool runs again once, and there is one `ToolCallEnd`.
async fn s2_safe_tool(backend: Backend) -> Result<()> {
    let Some(case) = crash_and_recover(backend, Scenario::SafeTool).await? else {
        return Ok(());
    };
    ensure!(
        case.runs == [SAFE_TOOL, SAFE_TOOL],
        "the safe tool should run once before the crash and once after: {:?}",
        case.runs
    );
    let ends = tool_call_ends(&thread_events(&case.stores, &case.thread).await?);
    let [end] = ends.as_slice() else {
        bail!("expected one ToolCallEnd, got {ends:?}");
    };
    ensure!(end.success, "the re-run safe call did not succeed: {end:?}");
    ensure!(
        case.root.status == TaskStatus::Completed,
        "the root turn is {:?}: {:?}",
        case.root.status,
        case.root.last_error
    );
    let live = live_rows(&case.stores, &case.thread).await?;
    ensure!(live.is_empty(), "rows left live: {live:?}");
    Ok(())
}

/// S3's request states: the crashed attempt was closed when the re-acquired
/// turn salvaged its LLM request, and the successor's request finished with
/// the turn.
async fn s3_requests_ended(
    case: &Recovered,
    attempts: &[agent_server::journal::TurnAttempt],
) -> Result<()> {
    // LLM request; the successor's request finished with the turn.
    let crashed: Vec<_> = attempts
        .iter()
        .filter(|attempt| attempt.closed_as_abandoned())
        .collect();
    ensure!(
        crashed.len() == 1,
        "expected the crashed attempt closed as abandoned: {attempts:?}"
    );
    let request = |attempt: &agent_server::journal::TurnAttempt| {
        agent_server::journal::effect::EffectKey::llm(&case.root.id, attempt.id.as_str())
    };
    let salvaged = agent_server::journal::DurableStore::effect(
        case.stores.task_store.as_ref(),
        &request(crashed[0]),
    )
    .await?
    .context("the crashed attempt's LLM request")?;
    ensure!(
        matches!(
            salvaged.state,
            agent_server::journal::effect::EffectState::Salvaged { .. }
        ),
        "the crashed request is {:?}",
        salvaged.state
    );
    let successor = attempts
        .iter()
        .filter(|attempt| !attempt.closed_as_abandoned())
        .max_by_key(|attempt| attempt.attempt_number)
        .context("the successor attempt")?;
    let completed = agent_server::journal::DurableStore::effect(
        case.stores.task_store.as_ref(),
        &request(successor),
    )
    .await?
    .context("the successor's LLM request")?;
    ensure!(
        completed.state == agent_server::journal::effect::EffectState::Completed,
        "the successor's request is {:?}",
        completed.state
    );
    Ok(())
}

/// S3: no duplicate assistant message, the partial text is either kept
/// as an aborted note or gone, and every attempt row is closed.
async fn s3_llm_stream(backend: Backend) -> Result<()> {
    let Some(case) = crash_and_recover(backend, Scenario::LlmStream).await? else {
        return Ok(());
    };
    ensure!(
        case.root.status == TaskStatus::Completed,
        "the root turn is {:?}: {:?}",
        case.root.status,
        case.root.last_error
    );
    let messages = case
        .stores
        .message_store
        .get(&case.thread)
        .await?
        .context("no transcript")?
        .messages;
    let texts = assistant_texts(&messages);
    let finals = texts
        .iter()
        .filter(|text| text.contains(FINAL_TEXT))
        .count();
    ensure!(
        finals == 1,
        "expected one final assistant message: {texts:?}"
    );
    let partials = texts
        .iter()
        .filter(|text| text.contains(PARTIAL_TEXT))
        .count();
    ensure!(
        partials == 1,
        "the cut-off text must be salvaged exactly once: {texts:?}"
    );
    // The salvaged response is followed by the note, before the final
    // answer.
    let partial_at = messages
        .iter()
        .position(|message| {
            message.role == llm::Role::Assistant
                && assistant_texts(std::slice::from_ref(message))
                    .iter()
                    .any(|text| text.contains(PARTIAL_TEXT))
        })
        .context("the salvaged response")?;
    ensure!(
        messages
            .get(partial_at + 1)
            .is_some_and(agent_server::journal::effect::is_aborted_response_note),
        "the salvaged response is not followed by the note: {messages:?}"
    );
    let attempts = case
        .stores
        .attempt_store
        .list_by_task(&case.root.id)
        .await?;
    let open: Vec<_> = attempts
        .iter()
        .filter(|attempt| attempt.outcome.is_none())
        .map(|attempt| attempt.attempt_number)
        .collect();
    ensure!(
        open.is_empty(),
        "attempt rows left open: {open:?} of {}",
        attempts.len()
    );
    s3_requests_ended(&case, &attempts).await?;
    let live = live_rows(&case.stores, &case.thread).await?;
    ensure!(live.is_empty(), "rows left live: {live:?}");
    Ok(())
}

/// S6 (question): the prompt is visible after the restart and the answer
/// continues the turn once.
async fn s6_question(backend: Backend) -> Result<()> {
    let Some(case) = crash_and_recover(backend, Scenario::Question).await? else {
        return Ok(());
    };
    ensure!(
        case.root.status == TaskStatus::Completed,
        "the root turn is {:?}: {:?}",
        case.root.status,
        case.root.last_error
    );
    let answered = case
        .requests
        .iter()
        .filter(|results| !results.is_empty())
        .count();
    ensure!(
        answered == 1,
        "the answer should resume the turn once: {:?}",
        case.requests
    );
    let live = live_rows(&case.stores, &case.thread).await?;
    ensure!(live.is_empty(), "rows left live: {live:?}");
    Ok(())
}

/// S6 (tool confirmation): the prompt is visible after the restart and
/// the approval runs the tool once.
async fn s6_confirmation(backend: Backend) -> Result<()> {
    confirmation_case(backend, Scenario::Confirmation).await
}

/// S6 (tool confirmation) on a `SingleHost` daemon. The approval must not
/// be served before the restarted daemon begins its epoch: a lease granted
/// first carries no epoch, so the boot reclaim would requeue the approved
/// child under its live drive and the turn would wedge (F4). The race is
/// narrow; run it with `--stress-count 32` to measure it.
async fn s6_confirmation_single_host(backend: Backend) -> Result<()> {
    confirmation_case(backend, Scenario::ConfirmationSingleHost).await
}

async fn confirmation_case(backend: Backend, scenario: Scenario) -> Result<()> {
    let Some(case) = crash_and_recover(backend, scenario).await? else {
        return Ok(());
    };
    ensure!(
        case.runs == [CONFIRM_TOOL],
        "the approved tool should run once: {:?}",
        case.runs
    );
    let ends = tool_call_ends(&thread_events(&case.stores, &case.thread).await?);
    ensure!(ends.len() == 1, "expected one ToolCallEnd, got {ends:?}");
    ensure!(
        case.root.status == TaskStatus::Completed,
        "the root turn is {:?}: {:?}",
        case.root.status,
        case.root.last_error
    );
    let live = live_rows(&case.stores, &case.thread).await?;
    ensure!(live.is_empty(), "rows left live: {live:?}");
    Ok(())
}

/// Poll `check` until it yields a value, for at most `wait`.
async fn poll<T, F, Fut>(what: &str, wait: Duration, mut check: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Option<T>>>,
{
    let deadline = tokio::time::Instant::now() + wait;
    loop {
        if let Some(value) = check().await? {
            return Ok(value);
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out after {wait:?} waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Every task row of every thread.
async fn all_rows(stores: &StoreRegistry) -> Result<Vec<AgentTask>> {
    let mut rows = Vec::new();
    for thread in stores.thread_store.list().await? {
        rows.extend(stores.task_store.list_by_thread(&thread.thread_id).await?);
    }
    Ok(rows)
}

fn is_deadlock(status: &tonic::Status) -> bool {
    let message = status.message().to_lowercase();
    message.contains("40p01") || message.contains("deadlock")
}

#[allow(dead_code)] // Called by an expected-failure branch; none is open today.
fn known_failure_enabled() -> bool {
    std::env::var_os(KNOWN_FAILURES_ENV).is_some()
}

/// S4: a fork killed after its commit and before its reply. A retry
/// with the same `request_id` returns the committed fork, and no other
/// thread exists.
#[cfg(feature = "failpoints")]
async fn s4_fork(backend: Backend) -> Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = Arc::new(CaseDir(temp.path().to_owned()));
    let Some(db) = CaseDb::create(backend, temp.path()).await? else {
        return Ok(());
    };
    let stores = inspection(&db.storage).await?;
    let mut crashed = CrashedHost::spawn(Scenario::Fork, &dir, &db.storage)?;
    let mut control = connect(&crashed.endpoint(&dir).await?).await?;
    let (source, root) = start_turn(&mut control).await?;
    wait_for_task(
        &stores,
        &AgentTaskId::from_string(&root),
        "completed",
        |task| task.status == TaskStatus::Completed,
    )
    .await?;
    let fork = pb::ForkThreadRequest {
        request_id: "cm-fork".into(),
        source_thread_id: source.clone(),
        fork_after_committed_turns: 1,
        thread_id: String::new(),
    };
    let mut forking = control.clone();
    let request = fork.clone();
    let pending = tokio::spawn(async move { forking.fork_thread(request).await });
    // The crash point is reached once the fork's thread is committed.
    let committed = poll("the fork commit", WAIT, || async {
        let threads = stores.thread_store.list().await?;
        Ok(threads
            .into_iter()
            .find(|thread| thread.thread_id.to_string() != source)
            .map(|thread| thread.thread_id))
    })
    .await?;
    // The paused fail point holds the reply; a fork that replies here
    // did not reach the crash point.
    tokio::time::sleep(Duration::from_millis(500)).await;
    ensure!(!pending.is_finished(), "the fork replied before the crash");
    drop(control);
    crashed.kill_nine()?;
    pending.abort();

    let daemon = start_daemon(db.storage.clone(), Scenario::Fork, Phase::Recover, dir).await?;
    let mut control = connect(&daemon.host.endpoint()).await?;
    let retried = control
        .fork_thread(fork.clone())
        .await
        .context("the fork retry failed")?
        .into_inner();
    let again = control.fork_thread(fork).await?.into_inner();
    let id_of = |response: &pb::ForkThreadResponse| {
        response
            .thread
            .as_ref()
            .and_then(|view| view.thread.as_ref())
            .map(|thread| thread.thread_id.clone())
    };
    ensure!(
        id_of(&retried) == Some(committed.to_string()),
        "the retry did not return the committed fork {committed}: {:?}",
        id_of(&retried)
    );
    ensure!(
        id_of(&again) == id_of(&retried),
        "a second retry forked again"
    );
    daemon.report(Scenario::Fork.name());
    let mut threads: Vec<String> = stores
        .thread_store
        .list()
        .await?
        .into_iter()
        .map(|thread| thread.thread_id.to_string())
        .collect();
    threads.sort();
    let mut expected = vec![source, committed.to_string()];
    expected.sort();
    ensure!(
        threads == expected,
        "orphan threads after the fork retry: {threads:?}"
    );
    ensure!(retried.message_count > 0, "the fork copied no messages");
    Ok(())
}

/// S5: a cancel of a root with tool children and a subagent, killed
/// `delay` after the cancel request. After the restart and a client
/// retry of the cancel, every row is terminal, no unsafe call ran twice,
/// and no cancel hit a deadlock.
async fn s5_cancel_tree_once(backend: Backend, delay: Duration) -> Result<Option<String>> {
    let temp = tempfile::tempdir()?;
    let dir = Arc::new(CaseDir(temp.path().to_owned()));
    let Some(db) = CaseDb::create(backend, temp.path()).await? else {
        return Ok(None);
    };
    let stores = inspection(&db.storage).await?;
    let mut crashed = CrashedHost::spawn(Scenario::CancelTree, &dir, &db.storage)?;
    let mut control = connect(&crashed.endpoint(&dir).await?).await?;
    let (thread, root) = start_turn(&mut control).await?;
    wait_file(&mut crashed.child, &dir.subagent_ready()).await?;
    poll("every tool child to start", WAIT, || async {
        Ok((dir.run_lines()?.len() >= CANCEL_TOOL_CALLS).then_some(()))
    })
    .await?;
    let cancel = pb::CancelTaskRequest {
        task_id: root.clone(),
        reason: "crash-matrix cancel".into(),
        thread_id: thread.clone(),
    };
    let mut cancelling = control.clone();
    let request = cancel.clone();
    let pending = tokio::spawn(async move { cancelling.cancel_task(request).await });
    tokio::time::sleep(delay).await;
    crashed.kill_nine()?;
    let first = pending.await?;
    if let Err(status) = &first {
        ensure!(!is_deadlock(status), "the cancel deadlocked: {status}");
    }
    drop(control);

    let daemon = start_daemon(
        db.storage.clone(),
        Scenario::CancelTree,
        Phase::Recover,
        dir.clone(),
    )
    .await?;
    let mut control = connect(&daemon.host.endpoint()).await?;
    // The client saw no reply (or saw one), and retries.
    if let Err(status) = control.cancel_task(cancel).await {
        ensure!(
            !is_deadlock(&status),
            "the cancel retry deadlocked: {status}"
        );
    }
    let root = AgentTaskId::from_string(&root);
    let root_row =
        wait_for_task(&stores, &root, "terminal", |task| task.status.is_terminal()).await?;
    let rows = poll("every row terminal", WAIT, || async {
        let rows = all_rows(&stores).await?;
        Ok(rows
            .iter()
            .all(|row| row.status.is_terminal())
            .then_some(rows))
    })
    .await
    .with_context(|| format!("rows left live after the cancel (kill {delay:?} after it)"))?;
    daemon.report(&format!("{} {delay:?}", Scenario::CancelTree.name()));
    let runs = dir.run_lines()?;
    ensure!(
        runs.iter().filter(|run| *run == UNSAFE_TOOL).count() == CANCEL_TOOL_CALLS,
        "an unsafe tool child ran again: {runs:?}"
    );
    let threads = stores.thread_store.list().await?;
    ensure!(
        threads.len() == 2,
        "expected the root thread and its subagent thread: {threads:?}"
    );
    Ok(Some(format!(
        "kill {delay:?} after the cancel: first cancel {}, root {:?}, {} rows",
        if first.is_ok() {
            "answered"
        } else {
            "unanswered"
        },
        root_row.status,
        rows.len()
    )))
}

async fn s5_cancel_tree(backend: Backend) -> Result<()> {
    for delay in [0, 20, 200].map(Duration::from_millis) {
        if let Some(outcome) = s5_cancel_tree_once(backend, delay).await? {
            println!("s5 {backend:?}: {outcome}");
        }
    }
    Ok(())
}

/// How long a silent child may run before D-5 ends its parent's wait.
const STALL_BUDGET: Duration = Duration::from_secs(10);

/// S7: a child that never returns while its lease keeps beating. The
/// parent must continue, with a typed partial result, once the child's
/// stall budget ends (D-5). No process is killed.
async fn s7_hung(backend: Backend, scenario: Scenario) -> Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = Arc::new(CaseDir(temp.path().to_owned()));
    let Some(db) = CaseDb::create(backend, temp.path()).await? else {
        return Ok(());
    };
    let stores = inspection(&db.storage).await?;
    let daemon = start_daemon(
        db.storage.clone(),
        scenario,
        Phase::Recover,
        Arc::clone(&dir),
    )
    .await?;
    let mut control = connect(&daemon.host.endpoint()).await?;
    let (_thread, root) = start_turn(&mut control).await?;
    let started = if scenario == Scenario::HungSubagent {
        dir.subagent_ready()
    } else {
        dir.ready()
    };
    poll("the hung child to start", WAIT, || async {
        Ok(started.exists().then_some(()))
    })
    .await?;
    let root = AgentTaskId::from_string(&root);
    let continued = wait_for_task_within(
        &stores,
        &root,
        "continued past its hung child",
        WAIT,
        |task| task.status.is_terminal(),
    )
    .await;
    daemon.report(scenario.name());
    let root_row = continued.with_context(|| {
        format!("the parent is still waiting {WAIT:?} after its child went silent (budget {STALL_BUDGET:?})")
    })?;
    ensure!(
        root_row.status == TaskStatus::Completed,
        "the parent ended {:?}, not with a partial result: {:?}",
        root_row.status,
        root_row.last_error
    );
    let results = daemon.provider.requests();
    ensure!(
        results.iter().flatten().any(|(_, is_error)| *is_error),
        "the model was not told the child stalled: {results:?}"
    );
    Ok(())
}

async fn s7_hung_tool(backend: Backend) -> Result<()> {
    s7_hung(backend, Scenario::HungTool).await
}

async fn s7_hung_subagent(backend: Backend) -> Result<()> {
    s7_hung(backend, Scenario::HungSubagent).await
}

/// S8 (Postgres): a new host starts while the old one still runs a turn,
/// then the old host is killed. While both live, the new host must not
/// take the old host's leases (no second run of either tool); after the
/// kill it reclaims them and the turn ends with nothing stuck.
#[cfg(feature = "postgres")]
async fn s8_overlap(backend: Backend) -> Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = Arc::new(CaseDir(temp.path().to_owned()));
    let Some(db) = CaseDb::create(backend, temp.path()).await? else {
        return Ok(());
    };
    let stores = inspection(&db.storage).await?;
    let mut old = CrashedHost::spawn(Scenario::Overlap, &dir, &db.storage)?;
    let mut control = connect(&old.endpoint(&dir).await?).await?;
    let (thread, root) = start_turn(&mut control).await?;
    poll("both tool children to start", WAIT, || async {
        Ok((dir.run_lines()?.len() >= 2).then_some(()))
    })
    .await?;
    let thread = ThreadId::from_string(&thread);
    let before: Vec<AgentTask> = live_rows(&stores, &thread)
        .await?
        .into_iter()
        .filter(|row| row.parent_id.is_some())
        .collect();

    // Started on its own task: a host that waits for a leader lock does
    // not return from its start until it leads.
    let new = tokio::spawn(start_daemon(
        db.storage.clone(),
        Scenario::Overlap,
        Phase::Recover,
        Arc::clone(&dir),
    ));
    // Three lease periods with both hosts live.
    tokio::time::sleep(Duration::from_secs(9)).await;
    let runs_while_overlapping = dir.run_lines()?;
    let mut taken = Vec::new();
    for child in &before {
        let now = stores
            .task_store
            .get(&child.id)
            .await?
            .context("child lost")?;
        if now.status != child.status || now.attempt != child.attempt {
            taken.push(format!(
                "{}: {:?} attempt {} -> {:?} attempt {}",
                child.id, child.status, child.attempt, now.status, now.attempt
            ));
        }
    }
    drop(control);
    old.kill_nine()?;
    let new = tokio::time::timeout(WAIT, new)
        .await
        .context("the new host did not start after the old one died")???;

    let root = AgentTaskId::from_string(&root);
    let root_row =
        wait_for_task(&stores, &root, "terminal", |task| task.status.is_terminal()).await?;
    tokio::time::sleep(Duration::from_secs(2)).await;
    let runs = dir.run_lines()?;
    new.report(Scenario::Overlap.name());
    drop(new);
    ensure!(
        runs_while_overlapping.len() == 2 && taken.is_empty(),
        "the new host took the live host's work while both ran: runs {runs_while_overlapping:?}, taken {taken:?}"
    );
    ensure!(
        runs.iter().filter(|run| *run == UNSAFE_TOOL).count() == 1,
        "the unsafe tool ran again: {runs:?}"
    );
    ensure!(
        root_row.status == TaskStatus::Completed,
        "the root is {:?}: {:?}",
        root_row.status,
        root_row.last_error
    );
    let live = live_rows(&stores, &thread).await?;
    ensure!(live.is_empty(), "rows left live: {live:?}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// The matrix: each scenario on each backend
// ─────────────────────────────────────────────────────────────────────

/// S9 (prim10 design §5.3, the SDK half): a kill -9 while a subagent
/// invocation's completion hook runs. The hook's `GitStep` record committed
/// before the hook ran, so after the restart the requeued invocation
/// continues that record: the hook runs once more, on attempt 2 in
/// `Execute` with `prior = Started`, the record finishes `Completed` in the
/// settle, and the parent gets one result and completes.
async fn s9_merge_back(backend: Backend) -> Result<()> {
    let Some(case) = crash_and_recover(backend, Scenario::MergeBack).await? else {
        return Ok(());
    };
    let invocation = case.tool_child.as_ref().context("no invocation id")?;
    ensure!(
        case.runs == ["merge_back attempt=2 mode=Execute prior=started"],
        "the recovering hook ran {:?}",
        case.runs
    );
    let record = case
        .stores
        .task_store
        .effect(&EffectKey::git(invocation, "merge_back"))
        .await?
        .context("no git-step record")?;
    ensure!(
        record.state == EffectState::Completed && record.attempt == 1,
        "the git-step record: {record:?}"
    );
    let completed = thread_events(&case.stores, &case.thread)
        .await?
        .into_iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::SubagentProgress {
                    completed: true,
                    ..
                }
            )
        })
        .count();
    ensure!(
        completed == 1,
        "the parent got {completed} completed results"
    );
    ensure!(
        case.root.status == TaskStatus::Completed,
        "the root turn is {:?}: {:?}",
        case.root.status,
        case.root.last_error
    );
    let live = live_rows(&case.stores, &case.thread).await?;
    ensure!(live.is_empty(), "rows left live: {live:?}");
    Ok(())
}

macro_rules! matrix {
    ($backend:ident, $feature:literal, [$($scenario:ident),* $(,)?], [$($gated:ident: $gate:literal),* $(,)?], [$($known:ident: $why:literal),* $(,)?]) => {
        #[cfg(feature = $feature)]
        mod $backend {
            $(
                #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
                async fn $scenario() -> anyhow::Result<()> {
                    super::$scenario(super::Backend::from_name(stringify!($backend))).await
                }
            )*
            $(
                #[cfg(feature = $gate)]
                #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
                async fn $gated() -> anyhow::Result<()> {
                    super::$gated(super::Backend::from_name(stringify!($backend))).await
                }
            )*
            $(
                #[ignore = $why]
                #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
                async fn $known() -> anyhow::Result<()> {
                    super::$known(super::Backend::from_name(stringify!($backend))).await
                }
            )*
        }
    };
}

matrix!(
    sqlite,
    "sqlite",
    [
        s1_unsafe_tool,
        s2_safe_tool,
        s3_llm_stream,
        s5_cancel_tree,
        s6_question,
        s6_confirmation,
        s6_confirmation_single_host,
        s7_hung_tool,
        s9_merge_back,
        s7_hung_subagent,
    ],
    [s4_fork: "failpoints"],
    [
    ]
);

matrix!(
    postgres,
    "postgres",
    [
        s1_unsafe_tool,
        s2_safe_tool,
        s3_llm_stream,
        s5_cancel_tree,
        s6_question,
        s6_confirmation,
        s6_confirmation_single_host,
        s7_hung_tool,
        s9_merge_back,
        s7_hung_subagent,
    ],
    [s4_fork: "failpoints", s8_overlap: "postgres"],
    [
    ]
);
