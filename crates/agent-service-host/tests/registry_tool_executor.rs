//! End-to-end integration tests for [`RegistryToolExecutor`].
//!
//! These tests wire real SDK tools (from `agent-sdk`) through the
//! executor and assert they run successfully, proving that:
//!
//! 1. The full built-in tool suite registers cleanly via
//!    [`agent_sdk::builtin_tools::register_builtin_tools`].
//! 2. Dispatch through the v2 [`ToolCallExecutor`] contract — not just
//!    the inline `Tool::execute` trait — produces the expected
//!    [`ToolResult`] for each family.
//!
//! We exercise the private `dispatch()` path for the single-tool and
//! full-suite cases because constructing a full `ToolTaskBootstrap`
//! fixture from outside `agent-server` is heavy and independently
//! covered by `agent-server`'s worker tests.  The dispatch path is
//! what the executor's [`ToolCallExecutor::execute_tool_call`]
//! delegates to after extracting the turn from the bootstrap, so
//! covering `dispatch` gives us the real runtime contract without the
//! fixture churn.
//!
//! Since `dispatch` is private to `RegistryToolExecutor`, we can't
//! call it from outside the crate — so these tests live under
//! `tests/` only for integration-style multi-crate composition and
//! instead invoke the executor through the public [`ToolCallExecutor`]
//! trait using a small inline bootstrap fixture built from public
//! fields on [`AgentTask`], [`TaskState`], and
//! [`ContinuationEnvelope`].
//!
//! [`ToolCallExecutor`]: agent_service_host::runtime::ToolCallExecutor

use std::collections::HashMap;
use std::sync::Arc;

use agent_sdk::builtin_tools::{BuiltinToolsConfig, register_builtin_tools};
use agent_sdk::todo::TodoState;
use agent_sdk::{
    AgentCapabilities, DynamicToolName, Environment, InMemoryFileSystem, Tool, ToolContext,
    ToolRegistry, ToolResult,
};
use agent_sdk_foundation::types::{
    AgentContinuation, AgentState, PendingToolCallInfo, TokenUsage, ToolTier,
};
use agent_sdk_foundation::{ContinuationEnvelope, ThreadId};
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, WorkerId};
use agent_server::worker::{ToolEventCollector, ToolTaskBootstrap};
use agent_server::{TaskKind, TaskState, TaskStatus};
use agent_service_host::registry_tool_executor::RegistryToolExecutor;
use agent_service_host::runtime::ToolCallExecutor;
use serde_json::{Value, json};
use time::OffsetDateTime;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Build a minimal bootstrap fixture for the v2 executor trait.
///
/// Populates the fields the executor actually reads
/// (`thread_id`, `parent_task.id`, `parent_task.state`,
/// `tool_call.{name, input, effective_input}`); everything else gets
/// a sentinel default that's valid enough to satisfy the field types.
fn make_bootstrap(
    thread: &ThreadId,
    tool_name: &str,
    input: &serde_json::Value,
) -> ToolTaskBootstrap {
    make_bootstrap_with_context(thread, tool_name, input, input.clone(), HashMap::new())
}

fn make_bootstrap_with_context(
    thread: &ThreadId,
    tool_name: &str,
    input: &serde_json::Value,
    effective_input: serde_json::Value,
    metadata: HashMap<String, serde_json::Value>,
) -> ToolTaskBootstrap {
    let parent_id = AgentTaskId("parent-task".into());
    let child_id = AgentTaskId("child-task".into());
    let mut state = AgentState::new(thread.clone());
    state.metadata = metadata;

    let continuation = ContinuationEnvelope::wrap(AgentContinuation {
        thread_id: thread.clone(),
        turn: 7,
        total_usage: TokenUsage::default(),
        turn_usage: TokenUsage::default(),
        pending_tool_calls: Vec::new(),
        awaiting_index: 0,
        completed_results: Vec::new(),
        state,
        response_id: None,
        stop_reason: None,
        response_content: Vec::new(),
    });

    let parent_task = AgentTask {
        id: parent_id.clone(),
        kind: TaskKind::RootTurn,
        status: TaskStatus::WaitingOnChildren,
        parent_id: None,
        root_id: parent_id.clone(),
        depth: 0,
        thread_id: thread.clone(),
        submitted_input: Vec::new(),
        caller_metadata: None,
        worker_id: None,
        lease_id: None,
        lease_expires_at: None,
        last_heartbeat_at: None,
        state: TaskState::WaitingOnChildren {
            continuation: Box::new(continuation),
            suspended_messages: Vec::new(),
            child_ids: vec![child_id.clone()],
        },
        attempt: 0,
        max_attempts: 3,
        last_error: None,
        pending_child_count: 1,
        spawn_index: None,
        result_payload: None,
        otel_traceparent: None,
        created_at: OffsetDateTime::now_utc(),
        updated_at: OffsetDateTime::now_utc(),
        completed_at: None,
    };

    let mut child_task = parent_task.clone();
    child_task.id = child_id.clone();
    child_task.kind = TaskKind::ToolRuntime;
    child_task.status = TaskStatus::Running;
    child_task.parent_id = Some(parent_id);
    child_task.depth = 1;
    child_task.state = TaskState::None;
    child_task.worker_id = Some(WorkerId("worker-1".into()));
    child_task.lease_id = Some(LeaseId("lease-1".into()));
    child_task.spawn_index = Some(0);
    child_task.pending_child_count = 0;

    ToolTaskBootstrap {
        child_task,
        parent_task,
        thread_id: thread.clone(),
        task_id: child_id,
        worker_id: WorkerId("worker-1".into()),
        lease_id: LeaseId("lease-1".into()),
        tool_call: PendingToolCallInfo {
            id: "call-1".into(),
            name: tool_name.into(),
            display_name: tool_name.into(),
            tier: ToolTier::Observe,
            input: input.clone(),
            effective_input,
            listen_context: None,
        },
    }
}

struct ContextEchoTool;

impl Tool<()> for ContextEchoTool {
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new("ctx_echo")
    }

    fn display_name(&self) -> &'static str {
        "Context Echo"
    }

    fn description(&self) -> &'static str {
        "Echoes input text and forwarded metadata"
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, ctx: &ToolContext<()>, input: Value) -> anyhow::Result<ToolResult> {
        let text = input
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("missing-text");
        let user_id = ctx
            .metadata
            .get("user_id")
            .and_then(Value::as_str)
            .unwrap_or("missing-user");
        Ok(ToolResult::success(format!("{text}:{user_id}")))
    }
}

fn executor_with_builtins() -> RegistryToolExecutor<()> {
    let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
    let todo = Arc::new(RwLock::new(TodoState::new()));
    let mut registry = ToolRegistry::<()>::new();
    register_builtin_tools(
        &mut registry,
        BuiltinToolsConfig {
            environment: fs,
            capabilities: AgentCapabilities::full_access(),
            todo_state: Some(todo),
            link_fetch: false,
        },
    );
    RegistryToolExecutor::with_default_factory(Arc::new(registry), ())
}

async fn seed_workspace_with_file() -> Arc<InMemoryFileSystem> {
    let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
    fs.write_file("notes.txt", "alpha\nbeta\ngamma")
        .await
        .expect("seed notes");
    fs
}

#[tokio::test]
async fn executes_read_tool_end_to_end() -> anyhow::Result<()> {
    let fs = seed_workspace_with_file().await;
    let mut registry = ToolRegistry::<()>::new();
    register_builtin_tools(
        &mut registry,
        BuiltinToolsConfig {
            environment: Arc::clone(&fs),
            capabilities: AgentCapabilities::full_access(),
            todo_state: None,
            link_fetch: false,
        },
    );
    let executor = RegistryToolExecutor::with_default_factory(Arc::new(registry), ());

    let thread = ThreadId::from_string("t-read");
    let bootstrap = make_bootstrap(&thread, "read", &json!({"path": "/workspace/notes.txt"}));

    let result = executor
        .execute_tool_call(
            &bootstrap,
            ToolEventCollector::new(),
            CancellationToken::new(),
        )
        .await?;

    assert!(result.success, "read tool should succeed");
    assert!(
        result.output.contains("alpha"),
        "output should contain file contents, got: {}",
        result.output,
    );
    Ok(())
}

#[tokio::test]
async fn executes_primitive_todo_and_tool_families_through_v2() -> anyhow::Result<()> {
    // Prime the FS so read has something to see.
    let fs = seed_workspace_with_file().await;
    let todo = Arc::new(RwLock::new(TodoState::new()));

    let mut registry = ToolRegistry::<()>::new();
    register_builtin_tools(
        &mut registry,
        BuiltinToolsConfig {
            environment: Arc::clone(&fs),
            capabilities: AgentCapabilities::full_access(),
            todo_state: Some(Arc::clone(&todo)),
            link_fetch: false,
        },
    );
    let executor = RegistryToolExecutor::with_default_factory(Arc::new(registry), ());
    let thread = ThreadId::from_string("t-full-suite");

    // 1) Primitive: read.
    let result = executor
        .execute_tool_call(
            &make_bootstrap(&thread, "read", &json!({"path": "/workspace/notes.txt"})),
            ToolEventCollector::new(),
            CancellationToken::new(),
        )
        .await?;
    assert!(result.success, "primitive read should succeed");

    // 2) Todo family: write, then read.
    let todos = json!({
        "todos": [
            {"content": "Ship v2", "status": "in_progress", "activeForm": "Shipping v2"}
        ]
    });
    let result = executor
        .execute_tool_call(
            &make_bootstrap(&thread, "todo_write", &todos),
            ToolEventCollector::new(),
            CancellationToken::new(),
        )
        .await?;
    assert!(result.success, "todo_write should succeed");

    let result = executor
        .execute_tool_call(
            &make_bootstrap(&thread, "todo_read", &json!({})),
            ToolEventCollector::new(),
            CancellationToken::new(),
        )
        .await?;
    assert!(result.success, "todo_read should succeed");
    assert!(
        result.output.contains("Ship v2"),
        "todo_read should surface the write, got: {}",
        result.output,
    );

    Ok(())
}

#[tokio::test]
async fn unknown_tool_name_returns_dispatch_error() {
    let executor = executor_with_builtins();
    let thread = ThreadId::from_string("t-unknown");
    let bootstrap = make_bootstrap(&thread, "does_not_exist", &json!({}));

    let err = executor
        .execute_tool_call(
            &bootstrap,
            ToolEventCollector::new(),
            CancellationToken::new(),
        )
        .await
        .expect_err("unknown tool must fail");
    assert!(err.to_string().contains("does_not_exist"));
}

#[tokio::test]
async fn execute_tool_call_uses_effective_input_and_parent_metadata() -> anyhow::Result<()> {
    let mut registry = ToolRegistry::<()>::new();
    registry.register(ContextEchoTool);
    let executor = RegistryToolExecutor::with_default_factory(Arc::new(registry), ());
    let thread = ThreadId::from_string("t-effective-input");
    let bootstrap = make_bootstrap_with_context(
        &thread,
        "ctx_echo",
        &json!({"text": "requested"}),
        json!({"text": "effective"}),
        HashMap::from([(String::from("user_id"), json!("u-42"))]),
    );

    let result = executor
        .execute_tool_call(
            &bootstrap,
            ToolEventCollector::new(),
            CancellationToken::new(),
        )
        .await?;

    assert!(result.success);
    assert_eq!(result.output, "effective:u-42");
    Ok(())
}
