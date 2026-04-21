//! Canonical v2 tool-runtime wiring for the full built-in SDK tool
//! suite.
//!
//! This example shows the host-side composition that was previously
//! missing: how to put every built-in SDK tool behind the durable
//! [`ToolCallExecutor`] contract that `agent-server`'s
//! `execute_tool_task` drives.
//!
//! It deliberately stops short of spinning up the full worker loop —
//! that requires a live LLM provider and persistent stores, which
//! would obscure the one thing this example is about: the
//! **registry → executor → dispatch** chain that replaces the legacy
//! inline `Tool::execute` path for SDK tools.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example with_builtin_tools_v2 -p agent-service-host
//! ```
//!
//! [`ToolCallExecutor`]: agent_service_host::runtime::ToolCallExecutor

use std::sync::Arc;

use agent_sdk::builtin_tools::{BuiltinToolsConfig, register_builtin_tools};
use agent_sdk::todo::TodoState;
use agent_sdk::{AgentCapabilities, Environment, InMemoryFileSystem, ToolRegistry};
use agent_sdk_core::types::{AgentContinuation, AgentState, PendingToolCallInfo, TokenUsage};
use agent_sdk_core::{ContinuationEnvelope, ThreadId, ToolTier};
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, WorkerId};
use agent_server::worker::{ToolEventCollector, ToolTaskBootstrap};
use agent_server::{TaskKind, TaskState, TaskStatus};
use agent_service_host::registry_tool_executor::RegistryToolExecutor;
use agent_service_host::runtime::ToolCallExecutor;
use serde_json::json;
use time::OffsetDateTime;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Build a shared workspace and register every built-in tool
    //    into a single ToolRegistry<()>.  In a real host, `Ctx` might
    //    carry a database handle or a user-scoped auth token.
    let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
    fs.write_file("README.md", "# v2 demo\n\nHello from the new runtime.")
        .await?;

    let todo_state = Arc::new(RwLock::new(TodoState::new()));

    let mut registry = ToolRegistry::<()>::new();
    register_builtin_tools(
        &mut registry,
        BuiltinToolsConfig {
            environment: Arc::clone(&fs),
            capabilities: AgentCapabilities::full_access(),
            todo_state: Some(Arc::clone(&todo_state)),
            link_fetch: false,
        },
    );

    // 2. Wrap the registry in a RegistryToolExecutor.  This is the
    //    type a host plugs into ExecutionRuntime as its
    //    Arc<dyn ToolCallExecutor>.
    let executor: Arc<dyn ToolCallExecutor> = Arc::new(RegistryToolExecutor::with_default_factory(
        Arc::new(registry),
        (),
    ));

    // 3. Drive a few tool calls through the v2 contract.  In a real
    //    host these come from execute_tool_task after the parent
    //    worker suspended on a batch of pending_tool_calls — the
    //    bootstrap fixture here short-circuits that pipeline for
    //    demonstration.
    let thread = ThreadId::from_string("t-demo");

    run_call(
        executor.as_ref(),
        &thread,
        "read",
        json!({"path": "/workspace/README.md"}),
        "read README.md",
    )
    .await?;

    run_call(
        executor.as_ref(),
        &thread,
        "todo_write",
        json!({
            "todos": [
                {
                    "content": "Wire subagent into v2",
                    "status": "pending",
                    "activeForm": "Wiring subagent into v2"
                }
            ]
        }),
        "seed TODO list",
    )
    .await?;

    run_call(
        executor.as_ref(),
        &thread,
        "todo_read",
        json!({}),
        "read back TODO list",
    )
    .await?;

    Ok(())
}

async fn run_call(
    executor: &dyn ToolCallExecutor,
    thread: &ThreadId,
    tool_name: &str,
    input: serde_json::Value,
    label: &str,
) -> anyhow::Result<()> {
    let bootstrap = make_bootstrap(thread, tool_name, input);
    let result = executor
        .execute_tool_call(
            &bootstrap,
            ToolEventCollector::new(),
            CancellationToken::new(),
        )
        .await?;

    println!("--- {label} ({tool_name}) ---");
    println!("success: {}", result.success);
    println!("{}\n", result.output);
    Ok(())
}

/// Minimal bootstrap fixture: the executor only reads
/// `thread_id`, `parent_task.state`, and `tool_call.{name, input}`,
/// so everything else gets a sentinel that's valid enough to satisfy
/// the field types.
fn make_bootstrap(
    thread: &ThreadId,
    tool_name: &str,
    input: serde_json::Value,
) -> ToolTaskBootstrap {
    let parent_id = AgentTaskId("parent-task".into());
    let child_id = AgentTaskId("child-task".into());
    let worker_id = WorkerId("worker-1".into());
    let lease_id = LeaseId("lease-1".into());

    let continuation = ContinuationEnvelope::wrap(AgentContinuation {
        thread_id: thread.clone(),
        turn: 1,
        total_usage: TokenUsage::default(),
        turn_usage: TokenUsage::default(),
        pending_tool_calls: Vec::new(),
        awaiting_index: 0,
        completed_results: Vec::new(),
        state: AgentState::new(thread.clone()),
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
    child_task.worker_id = Some(worker_id.clone());
    child_task.lease_id = Some(lease_id.clone());
    child_task.spawn_index = Some(0);
    child_task.pending_child_count = 0;

    ToolTaskBootstrap {
        child_task,
        parent_task,
        thread_id: thread.clone(),
        task_id: child_id,
        worker_id,
        lease_id,
        tool_call: PendingToolCallInfo {
            id: "call-1".into(),
            name: tool_name.into(),
            display_name: tool_name.into(),
            tier: ToolTier::Observe,
            input: input.clone(),
            effective_input: input,
            listen_context: None,
        },
    }
}
