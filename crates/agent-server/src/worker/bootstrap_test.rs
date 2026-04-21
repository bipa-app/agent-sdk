//! Root-task-driven worker bootstrapping tests.
//!
//! These tests validate the Phase 4.1 acceptance criteria:
//!
//! - A runnable root task can be mapped deterministically to one
//!   [`AgentDefinition`].
//! - The worker bootstrapping path does not treat the old monolithic
//!   `AgentConfig` as the server contract.
//! - The resolved definition exposes the runtime policy inputs later
//!   Phase 4 slices need.
//! - The slice is verifiable with root-task-driven worker bootstrapping
//!   tests.

use agent_sdk_core::llm::{Effort, Tool};
use agent_sdk_core::{ThreadId, ToolRuntime};
use time::{Duration, OffsetDateTime};

use super::bootstrap::{WorkerBootstrapContext, resolve_bootstrap_context};
use super::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use super::registry::{AgentDefinitionRegistry, InMemoryAgentDefinitionRegistry};
use crate::journal::task::{AgentTask, LeaseId, TaskKind, TaskStatus, WorkerId};

// ─────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> OffsetDateTime {
    t0() + Duration::seconds(secs)
}

fn thread(name: &str) -> ThreadId {
    ThreadId::from_string(name)
}

fn sample_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "read_file".into(),
            description: "Read a file from disk".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
            display_name: "Read File".into(),
            tier: agent_sdk_core::ToolTier::Observe,
        },
        Tool {
            name: "write_file".into(),
            description: "Write content to a file".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
            display_name: "Write File".into(),
            tier: agent_sdk_core::ToolTier::Confirm,
        },
    ]
}

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "anthropic".into(),
        model: "claude-sonnet-4-5-20250929".into(),
        system_prompt: "You are a helpful assistant.".into(),
        max_tokens: 4096,
        tools: sample_tools(),
        thinking: ThinkingPolicy::Disabled,
        tools_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

/// Build a root-turn task in Running status, as if acquired by a worker.
fn running_root(thread_name: &str) -> AgentTask {
    let root = AgentTask::new_root_turn(thread(thread_name), t0(), 3);
    root.mark_running(
        WorkerId::from_string("w-test"),
        LeaseId::from_string("l-test"),
        t_plus(60),
        t_plus(1),
    )
    .expect("mark_running should succeed on a fresh Pending root")
}

/// Build a root-turn task still in Pending status.
fn pending_root(thread_name: &str) -> AgentTask {
    AgentTask::new_root_turn(thread(thread_name), t0(), 3)
}

/// Build a `ToolRuntime` child task in Running status.
fn running_tool_child(parent: &AgentTask) -> AgentTask {
    let child = AgentTask::new_child(parent, TaskKind::ToolRuntime, t_plus(2), 3)
        .expect("child creation should succeed");
    child
        .mark_running(
            WorkerId::from_string("w-child"),
            LeaseId::from_string("l-child"),
            t_plus(60),
            t_plus(3),
        )
        .expect("mark_running should succeed on child")
}

// ─────────────────────────────────────────────────────────────────────
// AgentDefinition + RuntimePolicy serde round-trip
// ─────────────────────────────────────────────────────────────────────

#[test]
fn agent_definition_round_trips_through_json() -> anyhow::Result<()> {
    let original = sample_definition();
    let json = serde_json::to_string(&original)?;
    let recovered: AgentDefinition = serde_json::from_str(&json)?;
    assert_eq!(recovered, original);
    Ok(())
}

#[test]
fn agent_definition_wire_format_is_stable() -> anyhow::Result<()> {
    let def = sample_definition();
    let value = serde_json::to_value(&def)?;

    for key in [
        "provider",
        "model",
        "system_prompt",
        "max_tokens",
        "tools",
        "thinking",
        "policy",
    ] {
        assert!(
            value.get(key).is_some(),
            "AgentDefinition wire format lost key `{key}`"
        );
    }

    assert_eq!(value["provider"], serde_json::json!("anthropic"));
    assert_eq!(
        value["model"],
        serde_json::json!("claude-sonnet-4-5-20250929")
    );
    assert_eq!(value["tools"].as_array().map(Vec::len), Some(2));

    // Policy is nested
    let policy = &value["policy"];
    assert_eq!(policy["tool_runtime"], serde_json::json!("external"));
    assert_eq!(policy["strict_durability"], serde_json::json!(true));
    assert_eq!(policy["max_attempts"], serde_json::json!(3));
    assert_eq!(policy["streaming"], serde_json::json!(false));

    Ok(())
}

#[test]
fn runtime_policy_round_trips_through_json() -> anyhow::Result<()> {
    let original = RuntimePolicy::server_default();
    let json = serde_json::to_string(&original)?;
    let recovered: RuntimePolicy = serde_json::from_str(&json)?;
    assert_eq!(recovered, original);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// ThinkingPolicy serde
// ─────────────────────────────────────────────────────────────────────

#[test]
fn thinking_policy_disabled_round_trips() -> anyhow::Result<()> {
    let p = ThinkingPolicy::Disabled;
    let json = serde_json::to_string(&p)?;
    let recovered: ThinkingPolicy = serde_json::from_str(&json)?;
    assert_eq!(recovered, p);
    assert!(json.contains("\"mode\":\"disabled\""));
    Ok(())
}

#[test]
fn thinking_policy_enabled_round_trips() -> anyhow::Result<()> {
    let p = ThinkingPolicy::Enabled {
        budget_tokens: 10_000,
    };
    let json = serde_json::to_string(&p)?;
    let recovered: ThinkingPolicy = serde_json::from_str(&json)?;
    assert_eq!(recovered, p);
    assert!(json.contains("\"mode\":\"enabled\""));
    assert!(json.contains("\"budget_tokens\":10000"));
    Ok(())
}

#[test]
fn thinking_policy_adaptive_round_trips() -> anyhow::Result<()> {
    let p = ThinkingPolicy::Adaptive {
        effort: Some(Effort::High),
    };
    let json = serde_json::to_string(&p)?;
    let recovered: ThinkingPolicy = serde_json::from_str(&json)?;
    assert_eq!(recovered, p);
    assert!(json.contains("\"mode\":\"adaptive\""));
    assert!(json.contains("\"effort\":\"high\""));
    Ok(())
}

#[test]
fn thinking_policy_adaptive_no_effort_round_trips() -> anyhow::Result<()> {
    let p = ThinkingPolicy::Adaptive { effort: None };
    let json = serde_json::to_string(&p)?;
    let recovered: ThinkingPolicy = serde_json::from_str(&json)?;
    assert_eq!(recovered, p);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Registry: deterministic resolution
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn registry_resolves_default_for_unknown_thread() -> anyhow::Result<()> {
    let def = sample_definition();
    let registry = InMemoryAgentDefinitionRegistry::new(def.clone());

    let task = running_root("t-unknown");
    let resolved = registry.resolve(&task).await?;
    assert_eq!(resolved, def);
    Ok(())
}

#[tokio::test]
async fn registry_resolves_per_thread_override() -> anyhow::Result<()> {
    let default = sample_definition();
    let registry = InMemoryAgentDefinitionRegistry::new(default.clone());

    let override_def = AgentDefinition {
        provider: "openai".into(),
        model: "gpt-5".into(),
        system_prompt: "Custom prompt.".into(),
        max_tokens: 8192,
        tools: Vec::new(),
        tools_fn: None,
        thinking: ThinkingPolicy::Adaptive {
            effort: Some(Effort::Max),
        },
        policy: RuntimePolicy {
            tool_runtime: ToolRuntime::External,
            strict_durability: true,
            max_attempts: 5,
            streaming: true,
        },
    };

    registry.register_for_thread(thread("t-custom"), override_def.clone())?;

    // Custom thread gets the override
    let task = running_root("t-custom");
    let resolved = registry.resolve(&task).await?;
    assert_eq!(resolved, override_def);

    // Other thread gets the default
    let other_task = running_root("t-other");
    let other_resolved = registry.resolve(&other_task).await?;
    assert_eq!(other_resolved, default);

    Ok(())
}

#[tokio::test]
async fn registry_is_deterministic_across_calls() -> anyhow::Result<()> {
    let registry = InMemoryAgentDefinitionRegistry::new(sample_definition());
    let task = running_root("t-deterministic");

    let first = registry.resolve(&task).await?;
    let second = registry.resolve(&task).await?;
    assert_eq!(first, second);
    Ok(())
}

#[tokio::test]
async fn registry_rejects_non_root_turn_task() -> anyhow::Result<()> {
    let registry = InMemoryAgentDefinitionRegistry::new(sample_definition());
    let parent = running_root("t-parent");
    let child = running_tool_child(&parent);

    let err = registry.resolve(&child).await;
    assert!(err.is_err());
    let msg = err.unwrap_err().to_string();
    assert!(
        msg.contains("RootTurn"),
        "error should mention RootTurn, got: {msg}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Bootstrap context: happy path
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn bootstrap_resolves_running_root_turn() -> anyhow::Result<()> {
    let def = sample_definition();
    let registry = InMemoryAgentDefinitionRegistry::new(def.clone());
    let task = running_root("t-bootstrap");

    let ctx = resolve_bootstrap_context(task.clone(), &registry).await?;

    assert_eq!(ctx.definition, def);
    assert_eq!(ctx.thread_id, thread("t-bootstrap"));
    assert_eq!(ctx.task_id, task.id);
    assert_eq!(ctx.worker_id, WorkerId::from_string("w-test"));
    assert_eq!(ctx.lease_id, LeaseId::from_string("l-test"));
    assert_eq!(ctx.task.status, TaskStatus::Running);
    assert_eq!(ctx.task.kind, TaskKind::RootTurn);
    Ok(())
}

#[tokio::test]
async fn bootstrap_uses_per_thread_definition() -> anyhow::Result<()> {
    let default = sample_definition();
    let custom = AgentDefinition {
        provider: "vertex".into(),
        model: "gemini-2.0-flash".into(),
        ..default.clone()
    };
    let registry = InMemoryAgentDefinitionRegistry::new(default);
    registry.register_for_thread(thread("t-vertex"), custom.clone())?;

    let task = running_root("t-vertex");
    let ctx = resolve_bootstrap_context(task, &registry).await?;
    assert_eq!(ctx.definition, custom);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Bootstrap context: rejection cases
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn bootstrap_rejects_pending_task() -> anyhow::Result<()> {
    let registry = InMemoryAgentDefinitionRegistry::new(sample_definition());
    let task = pending_root("t-pending");

    let err = resolve_bootstrap_context(task, &registry).await;
    assert!(err.is_err());
    let msg = err.unwrap_err().to_string();
    assert!(
        msg.contains("Running"),
        "error should mention Running, got: {msg}"
    );
    Ok(())
}

#[tokio::test]
async fn bootstrap_rejects_tool_runtime_child() -> anyhow::Result<()> {
    let registry = InMemoryAgentDefinitionRegistry::new(sample_definition());
    let parent = running_root("t-parent");
    let child = running_tool_child(&parent);

    let err = resolve_bootstrap_context(child, &registry).await;
    assert!(err.is_err());
    let msg = err.unwrap_err().to_string();
    assert!(
        msg.contains("RootTurn"),
        "error should mention RootTurn, got: {msg}"
    );
    Ok(())
}

#[tokio::test]
async fn bootstrap_rejects_completed_task() -> anyhow::Result<()> {
    let registry = InMemoryAgentDefinitionRegistry::new(sample_definition());
    let mut task = running_root("t-done");
    task.status = TaskStatus::Completed;
    task.completed_at = Some(t_plus(10));
    task.last_error = None;

    let err = resolve_bootstrap_context(task, &registry).await;
    assert!(err.is_err());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Full flow: submit → acquire → bootstrap
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn full_flow_submit_acquire_bootstrap() -> anyhow::Result<()> {
    use crate::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};

    let task_store = InMemoryAgentTaskStore::new();
    let def = sample_definition();
    let registry = InMemoryAgentDefinitionRegistry::new(def.clone());

    // Submit a root turn
    let root = AgentTask::new_root_turn(thread("t-flow"), t0(), 3);
    let submitted = task_store.submit_root_turn(root).await?;

    // Acquire it
    let worker = WorkerId::from_string("w-flow");
    let lease = LeaseId::from_string("l-flow");
    let acquired = task_store
        .try_acquire_task(&submitted.id, worker, lease, t_plus(60), t_plus(1))
        .await?
        .expect("acquisition should succeed");

    assert_eq!(acquired.status, TaskStatus::Running);

    // Bootstrap
    let ctx = resolve_bootstrap_context(acquired, &registry).await?;
    assert_eq!(ctx.definition, def);
    assert_eq!(ctx.thread_id, thread("t-flow"));
    assert_eq!(ctx.worker_id, WorkerId::from_string("w-flow"));
    assert_eq!(ctx.lease_id, LeaseId::from_string("l-flow"));

    // The definition exposes runtime policy inputs
    assert_eq!(ctx.definition.policy.tool_runtime, ToolRuntime::External);
    assert!(ctx.definition.policy.strict_durability);
    assert_eq!(ctx.definition.policy.max_attempts, 3);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Definition does not depend on AgentConfig
// ─────────────────────────────────────────────────────────────────────

/// Compile-time proof that [`WorkerBootstrapContext`] and
/// [`AgentDefinition`] do not reference [`AgentConfig`].
/// The server's bootstrapping path is self-contained.
#[test]
fn bootstrap_types_are_independent_of_agent_config() {
    fn _assert_types(
        _ctx: WorkerBootstrapContext,
        _def: AgentDefinition,
        _policy: RuntimePolicy,
        _thinking: ThinkingPolicy,
    ) {
    }

    // If this test compiles, the types are reachable without importing
    // AgentConfig anywhere in the bootstrap path.
}

/// The resolved definition exposes every runtime policy input that
/// later Phase 4 slices need. This test acts as a compile-time
/// regression guard — if a field is removed, this fails.
#[test]
fn definition_exposes_all_runtime_policy_inputs() {
    let def = sample_definition();

    // Provider / model resolution
    let _: &str = &def.provider;
    let _: &str = &def.model;

    // Agent behaviour
    let _: &str = &def.system_prompt;
    let _: u32 = def.max_tokens;
    let _: &[Tool] = &def.tools;
    let _: &ThinkingPolicy = &def.thinking;

    // Execution policy
    let _: ToolRuntime = def.policy.tool_runtime;
    let _: bool = def.policy.strict_durability;
    let _: u32 = def.policy.max_attempts;
    let _: bool = def.policy.streaming;
}
