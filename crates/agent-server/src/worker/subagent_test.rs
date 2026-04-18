//! Durable subagent spawn contract tests.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;

use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::{
    AgentTask, AgentTaskStore, InMemoryAgentTaskStore, InMemoryThreadStore, LeaseId,
    SubagentInvocationSpawn, SuspensionPayload, TaskKind, TaskStatus, ThreadStore, WorkerId,
};
use agent_sdk_core::ToolTier;
use agent_sdk_core::audit::AuditProvenance;
use agent_sdk_core::events::AgentEvent;
use time::{Duration, OffsetDateTime};

use super::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
    InheritedSubagentConstraints, InheritedSubagentPolicy, ServerSubagentSpawnPolicy,
    SpawnedSubagentInvocation, SubagentCapabilityProfile, SubagentCapabilityRequest,
    SubagentInvocationDeps, SubagentMcpRequest, SubagentSandboxPolicy, SubagentSpawnPolicy,
    SubagentSpawnRequest, resolve_subagent_spec, spawn_subagent_invocation,
};

fn set(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn error_text(error: &anyhow::Error) -> String {
    format!("{error:#}")
}

struct FailingEventRepository;

#[async_trait]
impl EventRepository for FailingEventRepository {
    async fn commit_event(
        &self,
        _thread_id: &agent_sdk_core::ThreadId,
        _event: AgentEvent,
        _now: OffsetDateTime,
    ) -> Result<crate::journal::CommittedEvent> {
        anyhow::bail!("synthetic event commit failure");
    }

    async fn commit_event_batch(
        &self,
        _thread_id: &agent_sdk_core::ThreadId,
        _events: Vec<AgentEvent>,
        _now: OffsetDateTime,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        anyhow::bail!("synthetic event batch commit failure");
    }

    async fn next_sequence(&self, _thread_id: &agent_sdk_core::ThreadId) -> Result<u64> {
        Ok(0)
    }

    async fn get_events(
        &self,
        _thread_id: &agent_sdk_core::ThreadId,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        Ok(Vec::new())
    }

    async fn get_events_in_range(
        &self,
        _thread_id: &agent_sdk_core::ThreadId,
        _after_sequence: u64,
        _up_to_sequence: u64,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        Ok(Vec::new())
    }

    async fn threads_with_events_before(
        &self,
        _cutoff: OffsetDateTime,
        _limit: u32,
    ) -> Result<Vec<agent_sdk_core::ThreadId>> {
        Ok(Vec::new())
    }

    async fn max_sequence_before(
        &self,
        _thread_id: &agent_sdk_core::ThreadId,
        _cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        Ok(None)
    }

    async fn min_sequence_at_or_after(
        &self,
        _thread_id: &agent_sdk_core::ThreadId,
        _cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        Ok(None)
    }
}

fn sample_policy() -> InheritedSubagentPolicy {
    InheritedSubagentPolicy {
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
                    sandbox: SubagentSandboxPolicy::read_only().with_network_access(true),
                    allowed_mcp_servers: set(&["docs", "search"]),
                },
            ),
            (
                "edit".into(),
                SubagentCapabilityProfile {
                    capabilities: set(&["read_file", "rg", "apply_patch"]),
                    sandbox: SubagentSandboxPolicy::workspace_write(),
                    allowed_mcp_servers: set(&["docs"]),
                },
            ),
        ]),
        allowed_capabilities: set(&["read_file", "rg"]),
        max_depth: 3,
        max_parallel_subagents: 2,
        sandbox: SubagentSandboxPolicy::workspace_write(),
        allowed_mcp_servers: set(&["docs"]),
        audit_provider: "anthropic".into(),
    }
}

fn sample_constraints() -> InheritedSubagentConstraints {
    InheritedSubagentConstraints {
        policy: sample_policy(),
        current_depth: 0,
        active_parallel_subagents: 0,
    }
}

fn sample_spec(task: &str) -> EffectiveSubagentSpec {
    EffectiveSubagentSpec {
        task: task.into(),
        prompt: "Stay in read-only mode.".into(),
        model: "claude-sonnet-4-5-20250929".into(),
        max_turns: 5,
        timeout_ms: 15_000,
        depth: 1,
        max_parallel_subagents: 1,
        nickname: Some("Scout".into()),
        sandbox: SubagentSandboxPolicy::read_only(),
        mcp: EffectiveSubagentMcpPolicy {
            allowed_servers: set(&["docs"]),
        },
        audit_provenance: Some(AuditProvenance::new(
            "anthropic",
            "claude-sonnet-4-5-20250929",
        )),
        inherited_policy: InheritedSubagentPolicy {
            default_model: "claude-sonnet-4-5-20250929".into(),
            allowed_models: set(&["claude-sonnet-4-5-20250929"]),
            default_max_turns: 5,
            max_turns: 5,
            default_timeout_ms: 15_000,
            max_timeout_ms: 15_000,
            capability_profiles: sample_policy().capability_profiles,
            allowed_capabilities: set(&["read_file", "rg"]),
            max_depth: 3,
            max_parallel_subagents: 1,
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: set(&["docs"]),
            audit_provider: "anthropic".into(),
        },
        capabilities: EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: set(&["read_file", "rg"]),
        },
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
    .with_max_parallel_subagents(1)
    .with_sandbox(SubagentSandboxPolicy::read_only())
    .with_mcp_allowlist(["docs"])
    .with_nickname("Scout");

    let json = serde_json::to_string(&request)?;
    let round_trip: SubagentSpawnRequest = serde_json::from_str(&json)?;
    assert_eq!(round_trip, request);

    Ok(())
}

#[test]
fn effective_spec_round_trips_through_json() -> Result<()> {
    let spec = sample_spec("Summarize the storage contract");

    let json = serde_json::to_string(&spec)?;
    let round_trip: EffectiveSubagentSpec = serde_json::from_str(&json)?;
    assert_eq!(round_trip, spec);

    Ok(())
}

#[test]
fn legacy_effective_spec_without_inherited_policy_deserializes_fail_closed() -> Result<()> {
    let spec = sample_spec("Summarize the storage contract");
    let mut json = serde_json::to_value(&spec)?;
    let object = json
        .as_object_mut()
        .ok_or_else(|| anyhow!("expected effective spec JSON object"))?;
    object.remove("inherited_policy");
    object.remove("max_parallel_subagents");

    let legacy: EffectiveSubagentSpec = serde_json::from_value(json)?;
    let profile = legacy
        .inherited_policy
        .capability_profiles
        .get("research")
        .ok_or_else(|| anyhow!("expected derived legacy profile"))?;

    assert_eq!(
        legacy.inherited_policy.allowed_models,
        set(&[legacy.model.as_str()])
    );
    assert_eq!(
        legacy.inherited_policy.allowed_capabilities,
        legacy.capabilities.allowed
    );
    assert_eq!(legacy.inherited_policy.max_depth, legacy.depth);
    assert_eq!(legacy.max_parallel_subagents, 0);
    assert_eq!(legacy.inherited_policy.max_parallel_subagents, 1);
    assert_eq!(profile.capabilities, legacy.capabilities.allowed);
    assert_eq!(profile.sandbox, legacy.sandbox);
    assert_eq!(profile.allowed_mcp_servers, legacy.mcp.allowed_servers);

    let err = resolve_subagent_spec(
        &SubagentSpawnRequest::new(
            "Inspect durable bootstrap",
            SubagentCapabilityRequest::new("research"),
        ),
        &legacy.inherited_constraints(0),
        &ServerSubagentSpawnPolicy,
    )
    .err()
    .ok_or_else(|| anyhow!("expected legacy fallback constraints to fail closed"))?;
    assert!(
        error_text(&err).contains("depth limit exceeded"),
        "unexpected error: {err:#}"
    );

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
    assert_eq!(first.depth, 1);
    assert_eq!(first.max_parallel_subagents, 2);
    assert_eq!(first.nickname.as_deref(), Some("Scout"));
    assert_eq!(first.capabilities.allowed, set(&["read_file", "rg"]));
    assert_eq!(first.sandbox, SubagentSandboxPolicy::read_only());
    assert_eq!(first.mcp.allowed_servers, set(&["docs"]));
    assert_eq!(
        first.audit_provenance,
        Some(AuditProvenance::new(
            "anthropic",
            "claude-sonnet-4-5-20250929",
        )),
    );

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
    constraints.policy.allowed_capabilities = set(&["read_file", "does_not_exist"]);
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
    constraints.policy.capability_profiles.insert(
        "bad".into(),
        SubagentCapabilityProfile {
            capabilities: set(&["", "read_file"]),
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: BTreeSet::new(),
        },
    );
    constraints.policy.allowed_capabilities = set(&["", "read_file", "rg"]);

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
    constraints
        .policy
        .allowed_capabilities
        .insert(String::new());

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

struct AliasingPolicy;

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

    fn resolve_max_parallel_subagents(
        &self,
        _requested: Option<u32>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<u32> {
        Ok(1)
    }

    fn resolve_depth(&self, _constraints: &InheritedSubagentConstraints) -> Result<u32> {
        Ok(2)
    }

    fn resolve_sandbox(
        &self,
        _profile: &str,
        _requested: Option<&SubagentSandboxPolicy>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<SubagentSandboxPolicy> {
        Ok(SubagentSandboxPolicy::read_only())
    }

    fn resolve_mcp(
        &self,
        _profile: &str,
        _requested: Option<&SubagentMcpRequest>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentMcpPolicy> {
        Ok(EffectiveSubagentMcpPolicy {
            allowed_servers: set(&["docs"]),
        })
    }

    fn resolve_audit_provenance(
        &self,
        model: &str,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<AuditProvenance> {
        Ok(AuditProvenance::new("anthropic", model))
    }
}

impl SubagentSpawnPolicy for AliasingPolicy {
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
        requested: &SubagentCapabilityRequest,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentCapabilities> {
        ensure!(
            requested.profile == "research-lite",
            "expected alias input, got `{}`",
            requested.profile
        );
        Ok(EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: set(&["rg"]),
        })
    }

    fn resolve_max_parallel_subagents(
        &self,
        _requested: Option<u32>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<u32> {
        Ok(1)
    }

    fn resolve_depth(&self, _constraints: &InheritedSubagentConstraints) -> Result<u32> {
        Ok(2)
    }

    fn resolve_sandbox(
        &self,
        profile: &str,
        _requested: Option<&SubagentSandboxPolicy>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<SubagentSandboxPolicy> {
        ensure!(
            profile == "research",
            "sandbox received unresolved profile `{profile}`"
        );
        Ok(SubagentSandboxPolicy::read_only())
    }

    fn resolve_mcp(
        &self,
        profile: &str,
        _requested: Option<&SubagentMcpRequest>,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentMcpPolicy> {
        ensure!(
            profile == "research",
            "mcp received unresolved profile `{profile}`"
        );
        Ok(EffectiveSubagentMcpPolicy {
            allowed_servers: set(&["docs"]),
        })
    }

    fn resolve_audit_provenance(
        &self,
        model: &str,
        _constraints: &InheritedSubagentConstraints,
    ) -> Result<AuditProvenance> {
        Ok(AuditProvenance::new("anthropic", model))
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
    assert_eq!(spec.depth, 2);
    assert_eq!(spec.max_parallel_subagents, 1);
    assert_eq!(spec.capabilities.allowed, set(&["rg"]));
    assert_eq!(spec.mcp.allowed_servers, set(&["docs"]));

    Ok(())
}

#[test]
fn resolved_capability_profile_name_drives_sandbox_and_mcp_resolution() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("research-lite"),
    );

    let spec = resolve_subagent_spec(&request, &constraints, &AliasingPolicy)?;
    assert_eq!(spec.capabilities.profile, "research");
    assert_eq!(spec.sandbox, SubagentSandboxPolicy::read_only());
    assert_eq!(spec.mcp.allowed_servers, set(&["docs"]));

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
    parent_suspension_payload_with_tier(parent_thread_id, task, ToolTier::Confirm)
}

fn parent_suspension_payload_with_tier(
    parent_thread_id: &agent_sdk_core::ThreadId,
    task: &str,
    tier: ToolTier,
) -> SuspensionPayload {
    let tool_call = agent_sdk_core::PendingToolCallInfo {
        id: "call_subagent".into(),
        name: "subagent_researcher".into(),
        display_name: "Subagent: Researcher".into(),
        tier,
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
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let task = "Inspect durable linkage";
    let spec = sample_spec(task);

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
            event_repo: &event_repo,
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
    .await?;

    assert_eq!(created.committed_events.len(), 1);
    let parent_events = event_repo.get_events(&parent.thread_id).await?;
    assert_eq!(parent_events.len(), 1);
    let expected_child_root_task_id = created.child_root_task.id.to_string();
    let expected_subagent_task_id = created.invocation_task.id.to_string();
    match &parent_events[0].event {
        AgentEvent::SubagentProgress {
            subagent_name,
            child_thread_id,
            child_root_task_id,
            subagent_task_id,
            tool_name,
            completed,
            success,
            current_turn,
            tool_count,
            total_tokens,
            ..
        } => {
            assert_eq!(subagent_name, "researcher");
            assert_eq!(
                child_thread_id.as_ref(),
                Some(&created.child_thread.thread_id)
            );
            assert_eq!(
                child_root_task_id.as_deref(),
                Some(expected_child_root_task_id.as_str())
            );
            assert_eq!(
                subagent_task_id.as_deref(),
                Some(expected_subagent_task_id.as_str())
            );
            assert_eq!(tool_name, "researcher");
            assert!(!completed);
            assert!(!success);
            assert_eq!(*current_turn, Some(0));
            assert_eq!(*tool_count, 0);
            assert_eq!(*total_tokens, 0);
        }
        other => anyhow::bail!("expected SubagentProgress event, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn spawn_flow_tolerates_parent_progress_commit_failures() -> Result<()> {
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = FailingEventRepository;
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let task = "Inspect durable linkage";
    let spec = sample_spec(task);

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
            event_repo: &event_repo,
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
    .await?;
    assert!(created.committed_events.is_empty());

    Ok(())
}

#[test]
fn depth_limit_is_enforced() -> Result<()> {
    let mut constraints = sample_constraints();
    constraints.current_depth = constraints.policy.max_depth;

    let err = resolve_subagent_spec(
        &SubagentSpawnRequest::new(
            "Search for schema regressions",
            SubagentCapabilityRequest::new("research"),
        ),
        &constraints,
        &ServerSubagentSpawnPolicy,
    )
    .err()
    .ok_or_else(|| anyhow!("expected depth exhaustion to fail"))?;
    assert!(
        error_text(&err).contains("depth limit exceeded"),
        "unexpected error: {err:#}"
    );
    Ok(())
}

#[test]
fn parallel_budget_is_enforced() -> Result<()> {
    let mut constraints = sample_constraints();
    constraints.active_parallel_subagents = constraints.policy.max_parallel_subagents;

    let err = resolve_subagent_spec(
        &SubagentSpawnRequest::new(
            "Search for schema regressions",
            SubagentCapabilityRequest::new("research"),
        ),
        &constraints,
        &ServerSubagentSpawnPolicy,
    )
    .err()
    .ok_or_else(|| anyhow!("expected parallel budget exhaustion to fail"))?;
    assert!(
        error_text(&err).contains("parallel subagent budget exhausted"),
        "unexpected error: {err:#}"
    );
    Ok(())
}

#[test]
fn sandbox_request_cannot_widen_profile_or_parent() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("research"),
    )
    .with_sandbox(SubagentSandboxPolicy::full_access());

    let spec = resolve_subagent_spec(&request, &constraints, &ServerSubagentSpawnPolicy)?;
    assert_eq!(spec.sandbox, SubagentSandboxPolicy::read_only());
    Ok(())
}

#[test]
fn mcp_allowlist_narrows_profile_before_parent_ceiling() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("research"),
    )
    .with_mcp_allowlist(["search", "docs"]);

    let spec = resolve_subagent_spec(&request, &constraints, &ServerSubagentSpawnPolicy)?;
    assert_eq!(spec.mcp.allowed_servers, set(&["docs"]));
    Ok(())
}

#[test]
fn mcp_allowlist_cannot_widen_profile() -> Result<()> {
    let constraints = sample_constraints();
    let request = SubagentSpawnRequest::new(
        "Search for schema regressions",
        SubagentCapabilityRequest::new("edit"),
    )
    .with_mcp_allowlist(["search"]);

    let err = resolve_subagent_spec(&request, &constraints, &ServerSubagentSpawnPolicy)
        .err()
        .ok_or_else(|| anyhow!("expected MCP widening allowlist to fail"))?;
    assert!(
        error_text(&err).contains("MCP allowlist can only narrow profile servers"),
        "unexpected error: {err:#}"
    );
    Ok(())
}

#[test]
fn resolved_spec_carries_forward_nested_constraints() -> Result<()> {
    let constraints = sample_constraints();
    let spec = resolve_subagent_spec(
        &SubagentSpawnRequest::new(
            "Search for schema regressions",
            SubagentCapabilityRequest::new("research").with_allowlist(["rg"]),
        )
        .with_max_parallel_subagents(1)
        .with_mcp_allowlist(["docs"])
        .with_sandbox(SubagentSandboxPolicy::read_only()),
        &constraints,
        &ServerSubagentSpawnPolicy,
    )?;

    let nested = spec.inherited_constraints(0);
    assert_eq!(nested.current_depth, spec.depth);
    assert_eq!(nested.policy.allowed_models, set(&[spec.model.as_str()]));
    assert_eq!(nested.policy.max_turns, spec.max_turns);
    assert_eq!(nested.policy.max_timeout_ms, spec.timeout_ms);
    assert_eq!(nested.policy.allowed_capabilities, set(&["rg"]));
    assert_eq!(nested.policy.allowed_mcp_servers, set(&["docs"]));
    assert_eq!(nested.policy.max_parallel_subagents, 1);
    Ok(())
}

#[tokio::test]
async fn spawn_flow_rejects_non_confirm_tier_subagent_tools() -> Result<()> {
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let task = "Inspect durable linkage";

    let err = spawn_subagent_invocation(
        &parent.id,
        &worker,
        &lease,
        SubagentInvocationSpawn {
            child_thread_id: agent_sdk_core::ThreadId::new(),
            spec: sample_spec(task),
            child_root_input: child_root_input(task),
            spawn_index: 0,
            payload: parent_suspension_payload_with_tier(
                &parent.thread_id,
                task,
                ToolTier::Observe,
            ),
        },
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
            event_repo: &event_repo,
        },
        t_plus(2),
    )
    .await
    .err()
    .ok_or_else(|| anyhow!("expected observe-tier subagent spawn to fail"))?;
    assert!(
        error_text(&err).contains("must remain confirm-tier"),
        "unexpected error: {err:#}"
    );
    Ok(())
}
