//! Compile-time verification that the `agent-sdk` facade re-exports all
//! key types from the workspace sub-crates.
//!
//! Each function below names types from a specific layer. If a re-export
//! is accidentally dropped the test file will fail to compile — no runtime
//! assertions needed.

#![allow(
    clippy::items_after_statements,
    clippy::no_effect_underscore_binding,
    clippy::too_many_arguments,
    clippy::used_underscore_items
)]

// ── Core contract types (from agent-sdk-core) ────────────────────────

#[test]
fn core_ids_and_config() {
    use agent_sdk::{AgentConfig, AgentInput, RetryConfig, ThreadId};

    let _id = ThreadId::new();
    let _cfg = AgentConfig::default();
    let _retry = RetryConfig::default();
    let _input = AgentInput::Text("hi".into());
}

#[test]
fn core_events() {
    use agent_sdk::{AgentEvent, AgentEventEnvelope, SequenceCounter};

    let _seq = SequenceCounter::new();
    fn _assert(_e: AgentEvent, _env: AgentEventEnvelope) {}
}

#[test]
fn event_authority() {
    // Server-internal sequencing authority lives under `advanced`.
    use agent_sdk::advanced::{EventAuthority, LocalEventAuthority};

    let auth = LocalEventAuthority::new();
    let _: &dyn EventAuthority = &auth;

    let _seeded = LocalEventAuthority::with_offset(42);
}

#[test]
fn seed_and_factory_types() {
    // `ToolContextSeed` / `DefaultContextFactory` stay newcomer-facing;
    // the host-only factory trait + deps live under `advanced`.
    use agent_sdk::advanced::{ExecutionContextFactory, HostDependencies};
    use agent_sdk::{DefaultContextFactory, ToolContextSeed};

    let seed = ToolContextSeed::first_turn(agent_sdk::ThreadId::new());
    let _ = seed.with_metadata("key", serde_json::json!("val"));

    fn _assert_factory<F: ExecutionContextFactory<()>>(_f: &F) {}
    _assert_factory(&DefaultContextFactory);

    fn _assert_deps(_deps: HostDependencies) {}
}

#[test]
fn core_turn_outcomes() {
    // Server-facing turn outcome + continuation contract under `advanced`.
    use agent_sdk::advanced::{
        AgentContinuation, CONTINUATION_VERSION, ContinuationEnvelope, TurnOutcome,
    };
    use agent_sdk::{
        AgentError, AgentRunState, AgentState, ExecutionStatus, TokenUsage, ToolExecution,
        ToolInvocation, ToolOutcome, ToolResult, ToolRuntime, ToolTier, TurnOptions,
    };

    let _result = ToolResult::success("ok");
    let _usage = TokenUsage::default();
    let _ = CONTINUATION_VERSION;

    fn _assert(
        _c: AgentContinuation,
        _ce: ContinuationEnvelope,
        _e: AgentError,
        _rs: AgentRunState,
        _s: AgentState,
        _es: ExecutionStatus,
        _te: ToolExecution,
        _ti: ToolInvocation,
        _to: ToolOutcome,
        _turn: TurnOutcome,
        _tier: ToolTier,
        _opts: TurnOptions,
        _rt: ToolRuntime,
    ) {
    }
}

// ── Tool layer (from agent-sdk-tools) ────────────────────────────────

#[test]
fn tool_traits_and_registry() {
    use agent_sdk::{
        DynamicToolName, PrimitiveToolName, ProgressStage, ToolContext, ToolName, ToolRegistry,
        ToolStatus,
    };

    let _name = DynamicToolName::new("test");
    let _reg = ToolRegistry::<()>::new();
    let _ctx = ToolContext::new(());

    fn _assert_name<T: ToolName>() {}
    _assert_name::<DynamicToolName>();
    _assert_name::<PrimitiveToolName>();

    fn _assert_status<S: ProgressStage>(_s: ToolStatus<S>) {}
}

#[test]
fn hooks_and_stores() {
    use agent_sdk::{
        AgentHooks, AllowAllHooks, DefaultHooks, EventStore, InMemoryEventStore,
        InMemoryExecutionStore, InMemoryStore, LoggingHooks, MessageStore, StateStore,
        StoredTurnEvents, ToolDecision, ToolExecutionStore,
    };

    let _hooks: &dyn AgentHooks = &DefaultHooks;
    let _allow: &dyn AgentHooks = &AllowAllHooks;
    let _log: &dyn AgentHooks = &LoggingHooks;
    let _mem = InMemoryStore::new();
    let _events = InMemoryEventStore::new();
    let _exec = InMemoryExecutionStore::new();

    fn _assert_stores(
        _m: &dyn MessageStore,
        _s: &dyn StateStore,
        _e: &dyn EventStore,
        _t: &dyn ToolExecutionStore,
    ) {
    }
    fn _assert_decision(_d: ToolDecision) {}
    fn _assert_turns(_turns: StoredTurnEvents) {}
}

#[test]
fn audit_sink_and_records() {
    // The audit *sink* trait stays newcomer-facing (it's a builder hook);
    // the audit *record* protocol types live under `advanced`.
    use agent_sdk::advanced::{
        AuditProvenance, ToolAuditOutcome, ToolAuditRecord, ToolAuditRecordParams,
    };
    use agent_sdk::{NoopAuditSink, ToolAuditSink, ToolResult, ToolTier};

    // Trait object reachable from the facade.
    let _sink: &dyn ToolAuditSink = &NoopAuditSink;

    // Build a record so we cover every public constructor on the
    // facade re-export surface, including the discriminant accessor.
    let record = ToolAuditRecord::new(ToolAuditRecordParams {
        tool_call_id: "tool_call_id".into(),
        tool_name: "tool_name".into(),
        display_name: "Tool Display".into(),
        tier: ToolTier::Observe,
        requested_input: serde_json::json!({}),
        effective_input: serde_json::json!({}),
        turn: 1,
        provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5"),
        outcome: ToolAuditOutcome::Completed {
            result: ToolResult::success("ok"),
        },
    });
    assert_eq!(record.outcome_kind(), "completed");
}

#[test]
fn environment_types() {
    use agent_sdk::{Environment, ExecResult, FileEntry, GrepMatch, NullEnvironment};

    let _null = NullEnvironment;
    fn _assert(_env: &dyn Environment, _exec: ExecResult, _file: FileEntry, _grep: GrepMatch) {}
}

// ── Provider layer (from agent-sdk-providers) ────────────────────────

#[test]
fn provider_trait_and_streaming() {
    use agent_sdk::llm::{
        ChatOutcome, ChatRequest, ChatResponse, ContentBlock, ContentSource, Effort, LlmProvider,
        StreamAccumulator, StreamDelta, ThinkingConfig, ThinkingMode,
    };

    fn _assert_provider(_p: &dyn LlmProvider) {}
    fn _assert_streaming(_acc: StreamAccumulator, _delta: StreamDelta) {}
    fn _assert_types(
        _block: ContentBlock,
        _src: ContentSource,
        _effort: Effort,
        _tc: ThinkingConfig,
        _tm: ThinkingMode,
    ) {
    }
    fn _assert_chat(_req: ChatRequest, _res: ChatResponse, _out: ChatOutcome) {}
}

// Each provider re-export is gated behind its cargo feature, so the
// accessibility assertions are split per feature. Under `--all-features`
// every branch compiles; the default build only exercises `anthropic`.
#[cfg(feature = "anthropic")]
#[test]
fn anthropic_provider_accessible() {
    use agent_sdk::providers::AnthropicProvider;
    fn _assert(_a: AnthropicProvider) {}
}

#[cfg(feature = "openai")]
#[test]
fn openai_providers_accessible() {
    use agent_sdk::providers::{OpenAIProvider, OpenAIResponsesProvider};
    fn _assert(_o: OpenAIProvider, _or: OpenAIResponsesProvider) {}
}

#[cfg(feature = "openai-codex")]
#[test]
fn openai_codex_provider_accessible() {
    use agent_sdk::providers::OpenAICodexResponsesProvider;
    fn _assert(_oc: OpenAICodexResponsesProvider) {}
}

#[cfg(feature = "gemini")]
#[test]
fn gemini_provider_accessible() {
    use agent_sdk::providers::GeminiProvider;
    fn _assert(_g: GeminiProvider) {}
}

#[cfg(feature = "vertex")]
#[test]
fn vertex_provider_accessible() {
    use agent_sdk::providers::VertexProvider;
    fn _assert(_v: VertexProvider) {}
}

#[cfg(feature = "cloudflare")]
#[test]
fn cloudflare_provider_accessible() {
    use agent_sdk::providers::CloudflareAIGatewayProvider;
    fn _assert(_c: CloudflareAIGatewayProvider) {}
}

#[test]
fn model_capabilities_accessible() {
    use agent_sdk::{
        ModelCapabilities, PricePoint, Pricing, SourceStatus, get_model_capabilities,
        supported_model_capabilities,
    };

    let _caps = supported_model_capabilities();
    let _ = get_model_capabilities("anthropic", "claude-sonnet-4-20250514");
    fn _assert(_mc: ModelCapabilities, _pp: PricePoint, _pr: Pricing, _ss: SourceStatus) {}
}

// ── Facade-owned modules ─────────────────────────────────────────────

#[test]
fn agent_loop_builder() {
    use agent_sdk::{AgentCapabilities, AgentHandle, AgentLoop, builder};

    let _builder = builder::<()>();

    fn _assert<
        P: agent_sdk::LlmProvider,
        H: agent_sdk::AgentHooks,
        M: agent_sdk::MessageStore,
        S: agent_sdk::StateStore,
    >(
        _loop_: AgentLoop<(), P, H, M, S>,
        _handle: AgentHandle,
        _caps: AgentCapabilities,
    ) {
    }
}

#[test]
fn subagent_types() {
    use agent_sdk::{METADATA_MAX_SUBAGENT_DEPTH, METADATA_SUBAGENT_DEPTH};
    use agent_sdk::{SubagentConfig, SubagentFactory, SubagentTool};

    let _ = METADATA_SUBAGENT_DEPTH;
    let _ = METADATA_MAX_SUBAGENT_DEPTH;
    let _cfg = SubagentConfig::new("test");

    fn _uses_types<P: agent_sdk::LlmProvider>(
        _tool: SubagentTool<P>,
        _factory: SubagentFactory<P>,
    ) {
    }
}

#[test]
fn todo_types() {
    use agent_sdk::{TodoItem, TodoReadTool, TodoState, TodoStatus, TodoWriteTool};

    let _state = TodoState::new();
    let _status = TodoStatus::Pending;
    fn _assert(_item: TodoItem, _read: TodoReadTool, _write: TodoWriteTool) {}
}

#[test]
fn user_interaction_types() {
    use agent_sdk::{
        AskUserQuestionTool, ConfirmationRequest, ConfirmationResponse, QuestionOption,
        QuestionRequest, QuestionResponse,
    };

    fn _assert(
        _tool: AskUserQuestionTool,
        _cr: ConfirmationRequest,
        _cres: ConfirmationResponse,
        _qo: QuestionOption,
        _qr: QuestionRequest,
        _qres: QuestionResponse,
    ) {
    }
}

#[test]
fn reminder_types() {
    use agent_sdk::{
        ReminderConfig, ReminderTracker, ReminderTrigger, ToolReminder, append_reminder,
        wrap_reminder,
    };

    let _wrapped = wrap_reminder("test");
    let _config = ReminderConfig::new();

    fn _assert(_tracker: ReminderTracker, _trigger: ReminderTrigger, _tool: ToolReminder) {}
    fn _assert_append(result: &mut agent_sdk::ToolResult) {
        append_reminder(result, "reminder");
    }
}

#[test]
fn cancellation_token() {
    use agent_sdk::CancellationToken;
    let _token = CancellationToken::new();
}

#[test]
fn filesystem_types() {
    use agent_sdk::{InMemoryFileSystem, LocalFileSystem};
    fn _assert(_mem: InMemoryFileSystem, _local: LocalFileSystem) {}
}

// ── Newcomer ergonomics (Phase 12 · A) ───────────────────────────────

#[test]
fn prelude_brings_newcomer_surface() {
    // `use agent_sdk::prelude::*` must resolve the common building blocks.
    use agent_sdk::prelude::*;

    let _builder = builder::<()>();
    let _reg = ToolRegistry::<()>::new();
    let _ctx = ToolContext::new(());
    let _store = InMemoryEventStore::new();
    let _token = CancellationToken::new();
    let _name = DynamicToolName::new("x");
    let _cfg = AgentConfig::default();
    let _result = ToolResult::success("ok");
    let _tier = ToolTier::Observe;
    let _input = AgentInput::Text("hi".into());

    fn _assert_provider(_p: AnthropicProvider) {}
    fn _assert_event(_e: AgentEvent) {}
    fn _assert_tool<T: Tool<()>>() {}
    fn _assert_simple<T: SimpleTool<()>>() {}
}

#[test]
fn simple_tool_needs_no_tool_name() {
    use agent_sdk::{SimpleTool, ToolContext, ToolRegistry, ToolResult};
    use serde_json::{Value, json};

    // A first custom tool with no `ToolName` type and no `Tool` impl.
    struct EchoTool;

    impl SimpleTool<()> for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echo the input"
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        async fn execute(
            &self,
            _ctx: &ToolContext<()>,
            input: Value,
        ) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::success(input.to_string()))
        }
    }

    let mut reg = ToolRegistry::<()>::new();
    reg.register_simple(EchoTool);
    assert!(reg.get("echo").is_some());
}

// `try_from_env` must compile against each first-party provider and accept
// `impl Into<String>` keys; conventional env-var names are exposed as
// associated consts. Each provider's checks are gated behind its feature so
// the assertions still hold under any partial feature set.
#[cfg(feature = "anthropic")]
#[test]
fn anthropic_from_env_constructors_exist() {
    use agent_sdk::providers::AnthropicProvider;

    fn _assert() {
        let _ = AnthropicProvider::try_from_env();
        let _ = AnthropicProvider::sonnet("key-as-str");
        let _ = AnthropicProvider::sonnet(String::from("key-as-string"));
    }

    assert_eq!(AnthropicProvider::API_KEY_ENV, "ANTHROPIC_API_KEY");
}

#[cfg(feature = "openai")]
#[test]
fn openai_from_env_constructors_exist() {
    use agent_sdk::providers::OpenAIProvider;

    fn _assert() {
        let _ = OpenAIProvider::try_from_env();
    }

    assert_eq!(OpenAIProvider::API_KEY_ENV, "OPENAI_API_KEY");
}

#[cfg(feature = "gemini")]
#[test]
fn gemini_from_env_constructors_exist() {
    use agent_sdk::providers::GeminiProvider;

    fn _assert() {
        let _ = GeminiProvider::try_from_env();
    }

    assert_eq!(GeminiProvider::API_KEY_ENV, "GEMINI_API_KEY");
}
