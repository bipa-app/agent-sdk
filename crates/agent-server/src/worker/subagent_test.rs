//! Durable subagent spawn contract tests.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow};

use crate::journal::{
    AgentTask, AgentTaskStore, InMemoryAgentTaskStore, InMemoryThreadStore, LeaseId,
    SubagentInvocationSpawn, SuspensionPayload, TaskKind, TaskStatus, ThreadStore, WorkerId,
};
use time::{Duration, OffsetDateTime};

use super::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentSpec, InheritedSubagentConstraints,
    ServerSubagentSpawnPolicy, SpawnedSubagentInvocation, SubagentCapabilityProfile,
    SubagentCapabilityRequest, SubagentInvocationDeps, SubagentSpawnPolicy, SubagentSpawnRequest,
    resolve_subagent_spec, spawn_subagent_invocation,
};

fn set(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn error_text(error: &anyhow::Error) -> String {
    format!("{error:#}")
}

fn sample_constraints() -> InheritedSubagentConstraints {
    InheritedSubagentConstraints {
        default_model: "claude-sonnet-4-5-20250929".into(),
        allowed_models: set(&["claude-sonnet-4-5-20250929", "claude-opus-4-5-20250929"]),
        default_max_turns: 8,
        max_turns: 12,
        default_timeout_ms: 30_000,
        max_timeout_ms: 60_000,
        capability_profiles: BTreeMap::from([
            (
                "research".into(),
                SubagentCapabilityProfile {
                    capabilities: set(&["read_file", "rg", "web_search"]),
                },
            ),
            (
                "edit".into(),
                SubagentCapabilityProfile {
                    capabilities: set(&["read_file", "rg", "apply_patch"]),
                },
            ),
        ]),
        allowed_capabilities: set(&["read_file", "rg"]),
    }
}

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn t_plus(secs: i64) -> OffsetDateTime {
    t0() + Duration::seconds(secs)
}

#[test]
fn spawn_request_round_trips_through_json() -> Result<()> {
    let request = SubagentSpawnRequest::new(
        "Investigate retry drift",
        SubagentCapabilityRequest::new("research").with_allowlist(["read_file", "rg"]),
    )
    .with_prompt("Look for durable worker boundaries.")
    .with_model("claude-opus-4-5-20250929")
    .with_max_turns(10)
    .with_timeout_ms(45_000)
    .with_nickname("Scout");

    let json = serde_json::to_string(&request)?;
    let round_trip: SubagentSpawnRequest = serde_json::from_str(&json)?;
    assert_eq!(round_trip, request);

    Ok(())
}

#[test]
fn effective_spec_round_trips_through_json() -> Result<()> {
    let spec = EffectiveSubagentSpec {
        task: "Summarize the storage contract".into(),
        prompt: "Stay within the server boundary.".into(),
        model: "claude-sonnet-4-5-20250929".into(),
        max_turns: 6,
        timeout_ms: 20_000,
        nickname: Some("Scout".into()),
        capabilities: EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: set(&["read_file", "rg"]),
        },
    };

    let json = serde_json::to_string(&spec)?;
    let round_trip: EffectiveSubagentSpec = serde_json::from_str(&json)?;
    assert_eq!(round_trip, spec);

    Ok(())
}

#[test]
fn resolution_is_deterministic_for_same_inputs() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Find the durable bootstrap path",
        SubagentCapabilityRequest::new("research"),
    )
    .with_prompt("Focus on authoritative contracts.")
    .with_model("unsupported-model")
    .with_max_turns(99)
    .with_timeout_ms(120_000)
    .with_nickname("  Scout  ");
    let policy = ServerSubagentSpawnPolicy;

    let first = resolve_subagent_spec(&request, &constraints, &policy)?;
    let second = resolve_subagent_spec(&request, &constraints, &policy)?;
    assert_eq!(first, second);
    assert_eq!(first.model, "claude-sonnet-4-5-20250929");
    assert_eq!(first.max_turns, 12);
    assert_eq!(first.timeout_ms, 60_000);
    assert_eq!(first.nickname.as_deref(), Some("Scout"));
    assert_eq!(first.capabilities.allowed, set(&["read_file", "rg"]));

    Ok(())
}

#[test]
fn prompt_is_trimmed_and_blank_prompt_resolves_to_empty_string() -> Result<()> {
    let constraints = sample_constraints();
    let policy = ServerSubagentSpawnPolicy;

    let trimmed = resolve_subagent_spec(
        &SubagentSpawnRequest::new(
            "Find the durable bootstrap path",
            SubagentCapabilityRequest::new("research"),
        )
        .with_prompt("  Focus on authoritative contracts.  "),
        &constraints,
        &policy,
    )?;
    assert_eq!(trimmed.prompt, "Focus on authoritative contracts.");

    let blank = resolve_subagent_spec(
        &SubagentSpawnRequest::new(
            "Find the durable bootstrap path",
            SubagentCapabilityRequest::new("research"),
        )
        .with_prompt("   "),
        &constraints,
        &policy,
    )?;
    assert!(blank.prompt.is_empty());

    Ok(())
}

#[test]
fn allowlist_narrows_profile_before_parent_ceiling() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("research").with_allowlist(["rg"]),
    );
    let policy = ServerSubagentSpawnPolicy;

    let spec = resolve_subagent_spec(&request, &constraints, &policy)?;
    assert_eq!(spec.capabilities.profile, "research");
    assert_eq!(spec.capabilities.allowed, set(&["rg"]));

    Ok(())
}

#[test]
fn unknown_profile_is_rejected() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("unknown"),
    );
    let policy = ServerSubagentSpawnPolicy;

    let err = resolve_subagent_spec(&request, &constraints, &policy)
        .err()
        .ok_or_else(|| anyhow!("expected unknown profile to fail"))?;
    let message = error_text(&err);
    assert!(
        message.contains("unknown capability profile"),
        "unexpected error: {message}"
    );

    Ok(())
}

#[test]
fn allowlist_cannot_widen_profile() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("research").with_allowlist(["apply_patch"]),
    );
    let policy = ServerSubagentSpawnPolicy;

    let err = resolve_subagent_spec(&request, &constraints, &policy)
        .err()
        .ok_or_else(|| anyhow!("expected widening allowlist to fail"))?;
    let message = error_text(&err);
    assert!(
        message.contains("can only narrow profile"),
        "unexpected error: {message}"
    );

    Ok(())
}

#[test]
fn empty_task_is_rejected_before_policy_runs() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new("   ", SubagentCapabilityRequest::new("research"));
    let policy = ServerSubagentSpawnPolicy;

    let err = resolve_subagent_spec(&request, &constraints, &policy)
        .err()
        .ok_or_else(|| anyhow!("expected blank task to fail"))?;
    let message = error_text(&err);
    assert!(message.contains("task cannot be blank"));

    Ok(())
}

#[test]
fn inherited_constraints_validate_their_own_ceiling() -> Result<()> {
    let mut constraints = sample_constraints();
    constraints.allowed_capabilities = set(&["read_file", "does_not_exist"]);
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("research"),
    );
    let policy = ServerSubagentSpawnPolicy;

    let err = resolve_subagent_spec(&request, &constraints, &policy)
        .err()
        .ok_or_else(|| anyhow!("expected invalid ceiling to fail"))?;
    let message = error_text(&err);
    assert!(
        message.contains("allowed_capabilities contains unknown entries"),
        "unexpected error: {message}"
    );

    Ok(())
}

#[test]
fn task_is_trimmed_in_resolved_spec() -> Result<()> {
    let constraints = sample_constraints();
    let policy = ServerSubagentSpawnPolicy;
    let request =
        SubagentSpawnRequest::new("  run suite  ", SubagentCapabilityRequest::new("research"));

    let spec = resolve_subagent_spec(&request, &constraints, &policy)?;
    assert_eq!(spec.task, "run suite");

    Ok(())
}

#[test]
fn blank_model_is_rejected() -> Result<()> {
    let constraints = sample_constraints();
    let policy = ServerSubagentSpawnPolicy;
    let request = SubagentSpawnRequest::new(
        "Search for regressions",
        SubagentCapabilityRequest::new("research"),
    )
    .with_model("");

    let err = resolve_subagent_spec(&request, &constraints, &policy)
        .err()
        .ok_or_else(|| anyhow!("expected blank model to fail"))?;
    let message = error_text(&err);
    assert!(
        message.contains("model cannot be blank"),
        "unexpected error: {message}"
    );

    Ok(())
}

#[test]
fn whitespace_only_model_is_rejected() -> Result<()> {
    let constraints = sample_constraints();
    let policy = ServerSubagentSpawnPolicy;
    let request = SubagentSpawnRequest::new(
        "Search for regressions",
        SubagentCapabilityRequest::new("research"),
    )
    .with_model("   ");

    let err = resolve_subagent_spec(&request, &constraints, &policy)
        .err()
        .ok_or_else(|| anyhow!("expected blank model to fail"))?;
    let message = error_text(&err);
    assert!(
        message.contains("model cannot be blank"),
        "unexpected error: {message}"
    );

    Ok(())
}

#[test]
fn profile_name_is_trimmed_before_lookup() -> Result<()> {
    let constraints = sample_constraints();
    let policy = ServerSubagentSpawnPolicy;
    let request = SubagentSpawnRequest::new(
        "Search for regressions",
        SubagentCapabilityRequest::new(" research "),
    );

    let spec = resolve_subagent_spec(&request, &constraints, &policy)?;
    assert_eq!(spec.capabilities.profile, "research");

    Ok(())
}

#[test]
fn blank_capability_identifier_in_profile_is_rejected() -> Result<()> {
    let mut constraints = sample_constraints();
    constraints.capability_profiles.insert(
        "bad".into(),
        SubagentCapabilityProfile {
            capabilities: set(&["", "read_file"]),
        },
    );
    constraints.allowed_capabilities = set(&["", "read_file", "rg"]);

    let policy = ServerSubagentSpawnPolicy;
    let request = SubagentSpawnRequest::new(
        "Search for regressions",
        SubagentCapabilityRequest::new("research"),
    );

    let err = resolve_subagent_spec(&request, &constraints, &policy)
        .err()
        .ok_or_else(|| anyhow!("expected blank capability identifier to fail"))?;
    let message = error_text(&err);
    assert!(
        message.contains("capability identifier in profile")
            || message.contains("blank identifier"),
        "unexpected error: {message}"
    );

    Ok(())
}

#[test]
fn blank_allowed_capability_is_rejected() -> Result<()> {
    let mut constraints = sample_constraints();
    constraints.allowed_capabilities.insert(String::new());

    let policy = ServerSubagentSpawnPolicy;
    let request = SubagentSpawnRequest::new(
        "Search for regressions",
        SubagentCapabilityRequest::new("research"),
    );

    let err = resolve_subagent_spec(&request, &constraints, &policy)
        .err()
        .ok_or_else(|| anyhow!("expected blank allowed capability to fail"))?;
    let message = error_text(&err);
    assert!(
        message.contains("blank identifier"),
        "unexpected error: {message}"
    );

    Ok(())
}

struct FixedPolicy;

impl SubagentSpawnPolicy for FixedPolicy {
    fn resolve_model(
        &self,
        _requested: Option<&str>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<String> {
        Ok("gpt-5".into())
    }

    fn resolve_max_turns(
        &self,
        _requested: Option<u32>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<u32> {
        Ok(3)
    }

    fn resolve_timeout_ms(
        &self,
        _requested: Option<u64>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<u64> {
        Ok(5_000)
    }

    fn resolve_capabilities(
        &self,
        _requested: &SubagentCapabilityRequest,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentCapabilities> {
        Ok(EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: set(&["rg"]),
        })
    }
}

#[test]
fn custom_policy_hooks_drive_effective_resolution() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("research"),
    )
    .with_prompt("Focus on root-turn resume.");

    let spec = resolve_subagent_spec(&request, &constraints, &FixedPolicy)?;
    assert_eq!(spec.model, "gpt-5");
    assert_eq!(spec.max_turns, 3);
    assert_eq!(spec.timeout_ms, 5_000);
    assert_eq!(spec.capabilities.allowed, set(&["rg"]));

    Ok(())
}

async fn running_parent_root(
    task_store: &InMemoryAgentTaskStore,
) -> Result<(AgentTask, WorkerId, LeaseId)> {
    let root = AgentTask::new_root_turn(
        agent_sdk_core::ThreadId::from_string("t-parent-subagent"),
        t0(),
        3,
    );
    let parent_id = root.id.clone();
    task_store.submit_root_turn(root).await?;
    let worker = WorkerId::from_string("w-parent");
    let lease = LeaseId::from_string("l-parent");
    let claimed = task_store
        .try_acquire_task(
            &parent_id,
            worker.clone(),
            lease.clone(),
            t_plus(60),
            t_plus(1),
        )
        .await?
        .ok_or_else(|| anyhow!("expected parent root to be acquired"))?;
    Ok((claimed, worker, lease))
}

fn child_root_input(task: &str) -> Vec<crate::journal::task::SubmittedInputItem> {
    vec![crate::journal::task::SubmittedInputItem::Text {
        text: format!("Stay in read-only mode.\n\n{task}"),
    }]
}

fn parent_suspension_payload(
    parent_thread_id: &agent_sdk_core::ThreadId,
    task: &str,
) -> SuspensionPayload {
    let tool_call = agent_sdk_core::PendingToolCallInfo {
        id: "call_subagent".into(),
        name: "subagent_researcher".into(),
        display_name: "Subagent: Researcher".into(),
        tier: agent_sdk_core::ToolTier::Confirm,
        input: serde_json::json!({ "task": task }),
        effective_input: serde_json::json!({ "task": task }),
        listen_context: None,
    };
    SuspensionPayload {
        continuation: agent_sdk_core::ContinuationEnvelope::wrap(
            agent_sdk_core::AgentContinuation {
                thread_id: parent_thread_id.clone(),
                turn: 1,
                total_usage: agent_sdk_core::TokenUsage::default(),
                turn_usage: agent_sdk_core::TokenUsage::default(),
                pending_tool_calls: vec![tool_call],
                awaiting_index: 0,
                completed_results: Vec::new(),
                state: agent_sdk_core::AgentState::new(parent_thread_id.clone()),
                response_id: None,
                stop_reason: None,
                response_content: Vec::new(),
            },
        ),
        suspended_messages: Vec::new(),
    }
}

fn submitted_text(task: &AgentTask) -> Result<String> {
    task.submitted_input
        .iter()
        .map(|item| match item {
            crate::journal::task::SubmittedInputItem::Text { text } => Ok(text.clone()),
            other => anyhow::bail!("unexpected submitted input item: {other:?}"),
        })
        .collect::<Result<Vec<_>>>()
        .map(|parts| parts.join("\n"))
}

async fn assert_spawned_invocation_contract(
    task_store: &InMemoryAgentTaskStore,
    thread_store: &InMemoryThreadStore,
    parent: &AgentTask,
    created: &SpawnedSubagentInvocation,
    spec: &EffectiveSubagentSpec,
    expected_text: &str,
) -> Result<()> {
    assert_eq!(created.parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(created.parent_task.pending_child_count, 1);
    assert_eq!(created.invocation_task.kind, TaskKind::Subagent);
    assert_eq!(
        created.invocation_task.status,
        TaskStatus::WaitingOnChildren
    );
    assert_eq!(created.invocation_task.spawn_index, Some(0));
    assert_eq!(created.invocation_task.parent_id.as_ref(), Some(&parent.id));
    assert_eq!(created.invocation_task.thread_id, parent.thread_id);
    assert_eq!(created.child_root_task.kind, TaskKind::RootTurn);
    assert_eq!(created.child_root_task.status, TaskStatus::Pending);
    assert!(created.child_root_task.parent_id.is_none());
    assert_eq!(submitted_text(&created.child_root_task)?, expected_text);
    assert_eq!(
        created.child_thread.thread_id,
        created.child_root_task.thread_id
    );

    let linkage = created
        .invocation_task
        .state
        .subagent_invocation()
        .ok_or_else(|| anyhow!("invocation linkage missing"))?;
    assert_eq!(linkage.spec, *spec);
    assert_eq!(linkage.child_thread_id, created.child_thread.thread_id);
    assert_eq!(linkage.child_root_task_id, created.child_root_task.id);

    let parent_children = task_store.list_children(&parent.id).await?;
    assert_eq!(parent_children.len(), 1);
    assert_eq!(parent_children[0].id, created.invocation_task.id);

    let child_thread_tasks = task_store
        .list_by_thread(&created.child_thread.thread_id)
        .await?;
    assert_eq!(child_thread_tasks.len(), 1);
    assert_eq!(child_thread_tasks[0].id, created.child_root_task.id);

    let persisted_child_thread = thread_store
        .get(&created.child_thread.thread_id)
        .await?
        .ok_or_else(|| anyhow!("child thread projection missing"))?;
    assert_eq!(
        persisted_child_thread.thread_id,
        created.child_thread.thread_id
    );

    Ok(())
}

#[tokio::test]
async fn spawn_flow_creates_invocation_child_thread_and_child_root() -> Result<()> {
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let task = "Inspect durable linkage";
    let spec = EffectiveSubagentSpec {
        task: task.into(),
        prompt: "Stay in read-only mode.".into(),
        model: "claude-sonnet-4-5-20250929".into(),
        max_turns: 5,
        timeout_ms: 15_000,
        nickname: Some("Scout".into()),
        capabilities: EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: set(&["read_file", "rg"]),
        },
    };

    let child_thread_id = agent_sdk_core::ThreadId::new();
    let created: SpawnedSubagentInvocation = spawn_subagent_invocation(
        &parent.id,
        &worker,
        &lease,
        SubagentInvocationSpawn {
            child_thread_id,
            spec: spec.clone(),
            child_root_input: child_root_input(task),
            spawn_index: 0,
            payload: parent_suspension_payload(&parent.thread_id, task),
        },
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
        },
        t_plus(2),
    )
    .await?;

    assert_spawned_invocation_contract(
        &task_store,
        &thread_store,
        &parent,
        &created,
        &spec,
        "Stay in read-only mode.\n\nInspect durable linkage",
    )
    .await
}
