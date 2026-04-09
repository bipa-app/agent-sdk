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
    use agent_sdk::{EventAuthority, LocalEventAuthority};

    let auth = LocalEventAuthority::new();
    let _: &dyn EventAuthority = &auth;

    let _seeded = LocalEventAuthority::with_offset(42);
}

#[test]
fn seed_and_factory_types() {
    use agent_sdk::{
        DefaultContextFactory, ExecutionContextFactory, HostDependencies, ToolContextSeed,
    };

    let seed = ToolContextSeed::first_turn(agent_sdk::ThreadId::new());
    let _ = seed.with_metadata("key", serde_json::json!("val"));

    fn _assert_factory<F: ExecutionContextFactory<()>>(_f: &F) {}
    _assert_factory(&DefaultContextFactory);

    fn _assert_deps(_deps: HostDependencies) {}
}

#[test]
fn core_turn_outcomes() {
    use agent_sdk::{
        AgentContinuation, AgentError, AgentRunState, AgentState, CONTINUATION_VERSION,
        ContinuationEnvelope, ExecutionStatus, TokenUsage, ToolExecution, ToolInvocation,
        ToolOutcome, ToolResult, ToolRuntime, ToolTier, TurnOptions, TurnOutcome,
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
    use agent_sdk::{
        AuditProvenance, NoopAuditSink, ToolAuditOutcome, ToolAuditRecord, ToolAuditRecordParams,
        ToolAuditSink, ToolResult, ToolTier,
    };

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

#[test]
fn provider_impls_accessible() {
    use agent_sdk::providers::{
        AnthropicProvider, CloudflareAIGatewayProvider, GeminiProvider,
        OpenAICodexResponsesProvider, OpenAIProvider, OpenAIResponsesProvider, VertexProvider,
    };

    fn _assert(
        _a: AnthropicProvider,
        _c: CloudflareAIGatewayProvider,
        _g: GeminiProvider,
        _oc: OpenAICodexResponsesProvider,
        _o: OpenAIProvider,
        _or: OpenAIResponsesProvider,
        _v: VertexProvider,
    ) {
    }
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
