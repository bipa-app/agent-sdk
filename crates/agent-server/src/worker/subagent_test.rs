//! Durable subagent spawn contract tests.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;

use crate::journal::event_repository::{EventRepository, InMemoryEventRepository};
use crate::journal::{
    AgentTask, AgentTaskStore, InMemoryAgentTaskStore, InMemoryThreadStore, LeaseId,
    SubagentInvocationSpawn, SuspensionPayload, TaskKind, TaskStatus, ThreadStore, ToolChildSpawn,
    WorkerId,
};
use agent_sdk_foundation::ToolTier;
use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::events::AgentEvent;
use time::{Duration, OffsetDateTime};

use super::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
    InheritedSubagentConstraints, InheritedSubagentPolicy, MixedChildrenRequest,
    ServerSubagentSpawnPolicy, SpawnedSubagentBatch, SpawnedSubagentInvocation, SubagentBatchEntry,
    SubagentCapabilityProfile, SubagentCapabilityRequest, SubagentInvocationDeps,
    SubagentMcpRequest, SubagentSandboxPolicy, SubagentSpawnPolicy, SubagentSpawnRequest,
    resolve_subagent_spec, spawn_mixed_children_invocations, spawn_subagent_batch_invocations,
    spawn_subagent_invocation,
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
        _thread_id: &agent_sdk_foundation::ThreadId,
        _event: AgentEvent,
        _now: OffsetDateTime,
    ) -> Result<crate::journal::CommittedEvent> {
        anyhow::bail!("synthetic event commit failure");
    }

    async fn commit_event_batch(
        &self,
        _thread_id: &agent_sdk_foundation::ThreadId,
        _events: Vec<AgentEvent>,
        _now: OffsetDateTime,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        anyhow::bail!("synthetic event batch commit failure");
    }

    async fn next_sequence(&self, _thread_id: &agent_sdk_foundation::ThreadId) -> Result<u64> {
        Ok(0)
    }

    async fn get_events(
        &self,
        _thread_id: &agent_sdk_foundation::ThreadId,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        Ok(Vec::new())
    }

    async fn get_events_in_range(
        &self,
        _thread_id: &agent_sdk_foundation::ThreadId,
        _after_sequence: u64,
        _up_to_sequence: u64,
    ) -> Result<Vec<crate::journal::CommittedEvent>> {
        Ok(Vec::new())
    }

    async fn threads_with_events_before(
        &self,
        _cutoff: OffsetDateTime,
        _limit: u32,
    ) -> Result<Vec<agent_sdk_foundation::ThreadId>> {
        Ok(Vec::new())
    }

    async fn max_sequence_before(
        &self,
        _thread_id: &agent_sdk_foundation::ThreadId,
        _cutoff: OffsetDateTime,
    ) -> Result<Option<u64>> {
        Ok(None)
    }

    async fn min_sequence_at_or_after(
        &self,
        _thread_id: &agent_sdk_foundation::ThreadId,
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
        agent_sdk_foundation::ThreadId::from_string("t-parent-subagent"),
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
    parent_thread_id: &agent_sdk_foundation::ThreadId,
    task: &str,
) -> SuspensionPayload {
    parent_suspension_payload_with_tier(parent_thread_id, task, ToolTier::Confirm)
}

fn parent_suspension_payload_with_tier(
    parent_thread_id: &agent_sdk_foundation::ThreadId,
    task: &str,
    tier: ToolTier,
) -> SuspensionPayload {
    let tool_call = agent_sdk_foundation::PendingToolCallInfo {
        id: "call_subagent".into(),
        name: "subagent_researcher".into(),
        display_name: "Subagent: Researcher".into(),
        tier,
        input: serde_json::json!({ "task": task }),
        effective_input: serde_json::json!({ "task": task }),
        listen_context: None,
    };
    SuspensionPayload {
        continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
            agent_sdk_foundation::AgentContinuation {
                thread_id: parent_thread_id.clone(),
                turn: 1,
                total_usage: agent_sdk_foundation::TokenUsage::default(),
                turn_usage: agent_sdk_foundation::TokenUsage::default(),
                pending_tool_calls: vec![tool_call],
                awaiting_index: 0,
                completed_results: Vec::new(),
                state: agent_sdk_foundation::AgentState::new(parent_thread_id.clone()),
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

/// Build a `SuspensionPayload` with N pending tool calls (each a
/// distinct `subagent_*` tool, all Confirm-tier). Used to drive the
/// fan-out spawn helper, which validates each entry's `spawn_index`
/// against the shared envelope.
fn parent_suspension_payload_with_tools(
    parent_thread_id: &agent_sdk_foundation::ThreadId,
    tasks: &[&str],
) -> SuspensionPayload {
    let pending_tool_calls = tasks
        .iter()
        .enumerate()
        .map(|(idx, task)| agent_sdk_foundation::PendingToolCallInfo {
            id: format!("call_subagent_{idx}"),
            name: format!("subagent_researcher_{idx}"),
            display_name: format!("Subagent: Researcher {idx}"),
            tier: ToolTier::Confirm,
            input: serde_json::json!({ "task": task }),
            effective_input: serde_json::json!({ "task": task }),
            listen_context: None,
        })
        .collect();
    SuspensionPayload {
        continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
            agent_sdk_foundation::AgentContinuation {
                thread_id: parent_thread_id.clone(),
                turn: 1,
                total_usage: agent_sdk_foundation::TokenUsage::default(),
                turn_usage: agent_sdk_foundation::TokenUsage::default(),
                pending_tool_calls,
                awaiting_index: 0,
                completed_results: Vec::new(),
                state: agent_sdk_foundation::AgentState::new(parent_thread_id.clone()),
                response_id: None,
                stop_reason: None,
                response_content: Vec::new(),
            },
        ),
        suspended_messages: Vec::new(),
    }
}

#[tokio::test]
async fn spawn_batch_creates_n_invocations_under_one_parent() -> Result<()> {
    // Fan-out: a single CAS on the parent transitions it into
    // WaitingOnChildren with `pending_child_count = 3`, allocates
    // 3 invocation tasks + 3 child-thread root tasks, and emits
    // 3 SubagentProgress events on the parent thread.
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let tasks = vec!["explore A", "explore B", "explore C"];
    let payload = parent_suspension_payload_with_tools(&parent.thread_id, &tasks);

    let entries: Vec<SubagentBatchEntry> = tasks
        .iter()
        .enumerate()
        .map(|(idx, task)| SubagentBatchEntry {
            child_thread_id: agent_sdk_foundation::ThreadId::new(),
            spec: sample_spec(task),
            child_root_input: child_root_input(task),
            spawn_index: u32::try_from(idx).expect("test idx fits in u32"),
            child_caller_metadata: None,
        })
        .collect();

    let batch: SpawnedSubagentBatch = spawn_subagent_batch_invocations(
        &parent.id,
        &worker,
        &lease,
        entries,
        payload,
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
            event_repo: &event_repo,
        },
        t_plus(2),
    )
    .await?;

    assert_eq!(batch.parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(batch.parent_task.pending_child_count, 3);
    assert_eq!(batch.invocations.len(), 3);

    for (idx, invocation) in batch.invocations.iter().enumerate() {
        assert_eq!(invocation.invocation_task.kind, TaskKind::Subagent);
        assert_eq!(
            invocation.invocation_task.status,
            TaskStatus::WaitingOnChildren
        );
        assert_eq!(
            invocation.invocation_task.spawn_index,
            Some(u32::try_from(idx).expect("test idx fits in u32"))
        );
        assert_eq!(
            invocation.invocation_task.parent_id.as_ref(),
            Some(&parent.id)
        );
        assert_eq!(invocation.child_root_task.kind, TaskKind::RootTurn);
        assert_eq!(invocation.child_root_task.status, TaskStatus::Pending);
        assert_eq!(
            invocation.child_root_task.thread_id,
            invocation.child_thread.thread_id
        );
    }

    // Each invocation produced one SubagentProgress event on the
    // parent thread (3 total).
    let parent_events = event_repo.get_events(&parent.thread_id).await?;
    assert_eq!(parent_events.len(), 3);
    for event in &parent_events {
        match &event.event {
            AgentEvent::SubagentProgress {
                completed, success, ..
            } => {
                assert!(!completed);
                assert!(!success);
            }
            other => anyhow::bail!("expected SubagentProgress, got {other:?}"),
        }
    }

    Ok(())
}

#[tokio::test]
async fn spawn_batch_rejects_empty_input() -> Result<()> {
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    // Payload content is irrelevant here: the empty-batch guard fires
    // before the payload is inspected. Supply a well-formed one anyway.
    let payload = parent_suspension_payload_with_tools(&parent.thread_id, &["unused"]);

    let err = spawn_subagent_batch_invocations(
        &parent.id,
        &worker,
        &lease,
        Vec::new(),
        payload,
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
            event_repo: &event_repo,
        },
        t_plus(2),
    )
    .await
    .expect_err("empty batch should fail");
    assert!(
        format!("{err:#}").contains("subagent batch must be non-empty"),
        "unexpected: {err:#}"
    );

    Ok(())
}

#[tokio::test]
async fn spawn_batch_rejects_out_of_bounds_spawn_index() -> Result<()> {
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let tasks = vec!["A", "B"]; // 2 pending tool calls
    let payload = parent_suspension_payload_with_tools(&parent.thread_id, &tasks);

    let entries = vec![
        SubagentBatchEntry {
            child_thread_id: agent_sdk_foundation::ThreadId::new(),
            spec: sample_spec("A"),
            child_root_input: child_root_input("A"),
            spawn_index: 0,
            child_caller_metadata: None,
        },
        SubagentBatchEntry {
            child_thread_id: agent_sdk_foundation::ThreadId::new(),
            spec: sample_spec("ghost"),
            child_root_input: child_root_input("ghost"),
            spawn_index: 5, // out of bounds — only 2 pending tool calls
            child_caller_metadata: None,
        },
    ];

    let err = spawn_subagent_batch_invocations(
        &parent.id,
        &worker,
        &lease,
        entries,
        payload,
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
            event_repo: &event_repo,
        },
        t_plus(2),
    )
    .await
    .expect_err("out-of-bounds spawn_index should fail");
    assert!(
        format!("{err:#}").contains("out of bounds"),
        "unexpected: {err:#}"
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

    let child_thread_id = agent_sdk_foundation::ThreadId::new();
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
            child_caller_metadata: None,
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
async fn spawn_carries_child_caller_metadata_onto_child_root() -> Result<()> {
    // When the host supplies child_caller_metadata, it must land on the
    // child ROOT task so an AgentDefinitionRegistry can resolve a
    // per-subagent model/toolset from it. `None` (the default, covered
    // by every other test) leaves caller_metadata unset.
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let task = "Inspect durable linkage";
    let spec = sample_spec(task);
    let caller = serde_json::json!({ "subagent": "deep_research", "model": "claude-opus-4-8" });

    let created = spawn_subagent_invocation(
        &parent.id,
        &worker,
        &lease,
        SubagentInvocationSpawn {
            child_thread_id: agent_sdk_foundation::ThreadId::new(),
            spec: spec.clone(),
            child_root_input: child_root_input(task),
            spawn_index: 0,
            payload: parent_suspension_payload(&parent.thread_id, task),
            child_caller_metadata: Some(caller.clone()),
        },
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
            event_repo: &event_repo,
        },
        t_plus(2),
    )
    .await?;

    assert_eq!(
        created.child_root_task.caller_metadata.as_ref(),
        Some(&caller),
        "child root must carry the host-supplied caller metadata"
    );
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

    let child_thread_id = agent_sdk_foundation::ThreadId::new();
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
            child_caller_metadata: None,
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
            child_thread_id: agent_sdk_foundation::ThreadId::new(),
            spec: sample_spec(task),
            child_root_input: child_root_input(task),
            spawn_index: 0,
            payload: parent_suspension_payload_with_tier(
                &parent.thread_id,
                task,
                ToolTier::Observe,
            ),
            child_caller_metadata: None,
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

// Finding #20: a requested-but-disallowed model resolves to the
// inherited default (loud fallback) rather than erroring or being
// passed through.
#[test]
fn resolve_model_falls_back_to_default_for_disallowed_model() -> Result<()> {
    let constraints = sample_constraints();
    let policy = ServerSubagentSpawnPolicy;

    // A model outside the inherited allow-list resolves to the default.
    let resolved = policy.resolve_model(Some("gpt-4o-not-allowed"), &constraints)?;
    assert_eq!(resolved, "claude-sonnet-4-5-20250929");

    // An allowed model is preserved unchanged.
    let allowed = policy.resolve_model(Some("claude-opus-4-5-20250929"), &constraints)?;
    assert_eq!(allowed, "claude-opus-4-5-20250929");

    // No requested model → inherited default.
    let defaulted = policy.resolve_model(None, &constraints)?;
    assert_eq!(defaulted, "claude-sonnet-4-5-20250929");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Mixed batches: subagent spawns + tool children in one turn
// ─────────────────────────────────────────────────────────────────────

/// Build a payload whose pending tool calls mix subagent tools (the low
/// slots, Confirm-tier as the spawn path requires) with ordinary tool
/// calls (the high slots).
fn mixed_suspension_payload(
    parent_thread_id: &agent_sdk_foundation::ThreadId,
    subagent_tasks: &[&str],
    tool_names: &[&str],
) -> SuspensionPayload {
    let mut pending_tool_calls: Vec<agent_sdk_foundation::PendingToolCallInfo> = subagent_tasks
        .iter()
        .enumerate()
        .map(|(idx, task)| agent_sdk_foundation::PendingToolCallInfo {
            id: format!("call_subagent_{idx}"),
            name: format!("subagent_researcher_{idx}"),
            display_name: format!("Subagent: Researcher {idx}"),
            tier: ToolTier::Confirm,
            input: serde_json::json!({ "task": task }),
            effective_input: serde_json::json!({ "task": task }),
            listen_context: None,
        })
        .collect();
    pending_tool_calls.extend(tool_names.iter().enumerate().map(|(idx, name)| {
        agent_sdk_foundation::PendingToolCallInfo {
            id: format!("call_tool_{idx}"),
            name: (*name).to_owned(),
            display_name: (*name).to_owned(),
            tier: ToolTier::Observe,
            input: serde_json::json!({}),
            effective_input: serde_json::json!({}),
            listen_context: None,
        }
    }));
    SuspensionPayload {
        continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
            agent_sdk_foundation::AgentContinuation {
                thread_id: parent_thread_id.clone(),
                turn: 1,
                total_usage: agent_sdk_foundation::TokenUsage::default(),
                turn_usage: agent_sdk_foundation::TokenUsage::default(),
                pending_tool_calls,
                awaiting_index: 0,
                completed_results: Vec::new(),
                state: agent_sdk_foundation::AgentState::new(parent_thread_id.clone()),
                response_id: None,
                stop_reason: None,
                response_content: Vec::new(),
            },
        ),
        suspended_messages: Vec::new(),
    }
}

/// Subagent entries for the low slots, one per task, each on a fresh
/// child thread.
fn mixed_subagent_entries(tasks: &[&str]) -> Result<Vec<SubagentBatchEntry>> {
    tasks
        .iter()
        .enumerate()
        .map(|(idx, task)| {
            Ok(SubagentBatchEntry {
                child_thread_id: agent_sdk_foundation::ThreadId::new(),
                spec: sample_spec(task),
                child_root_input: child_root_input(task),
                spawn_index: u32::try_from(idx).map_err(|_| anyhow!("slot exceeds u32"))?,
                child_caller_metadata: None,
            })
        })
        .collect()
}

/// Tool-child entries for the slots `subagent_count..total`.
fn mixed_tool_children(subagent_count: usize, total: usize) -> Result<Vec<ToolChildSpawn>> {
    (subagent_count..total)
        .map(|slot| {
            Ok(ToolChildSpawn {
                spawn_index: u32::try_from(slot).map_err(|_| anyhow!("slot exceeds u32"))?,
                spec: crate::journal::ChildSpawnSpec::new(3),
            })
        })
        .collect()
}

#[tokio::test]
async fn spawn_mixed_creates_subagents_and_tool_children_under_one_parent() -> Result<()> {
    // One CAS on the parent persists BOTH halves of a mixed batch: two
    // durable subagent invocations (each with a child thread + child
    // root) and one tool-runtime child, with the parent parked on all
    // three. This is the turn shape an LLM coordinator routinely emits
    // (N subagent calls + a stray tool call).
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let tasks = ["explore A", "explore B"];
    let payload = mixed_suspension_payload(&parent.thread_id, &tasks, &["todo_write"]);

    let batch = spawn_mixed_children_invocations(
        &parent.id,
        &worker,
        &lease,
        MixedChildrenRequest {
            subagents: mixed_subagent_entries(&tasks)?,
            tool_children: mixed_tool_children(tasks.len(), 3)?,
            payload,
            child_otel_traceparent: None,
        },
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
            event_repo: &event_repo,
        },
        t_plus(2),
    )
    .await?;

    assert_eq!(batch.parent_task.status, TaskStatus::WaitingOnChildren);
    assert_eq!(batch.parent_task.pending_child_count, 3);
    assert_eq!(batch.invocations.len(), 2);
    assert_eq!(batch.tool_children.len(), 1);

    for (idx, invocation) in batch.invocations.iter().enumerate() {
        assert_eq!(invocation.invocation_task.kind, TaskKind::Subagent);
        assert_eq!(
            invocation.invocation_task.spawn_index,
            Some(u32::try_from(idx).map_err(|_| anyhow!("slot exceeds u32"))?)
        );
        assert_eq!(invocation.child_root_task.kind, TaskKind::RootTurn);
        assert_eq!(invocation.child_root_task.status, TaskStatus::Pending);
        assert_eq!(
            invocation.child_root_task.thread_id,
            invocation.child_thread.thread_id
        );
    }

    let tool_child = batch
        .tool_children
        .first()
        .ok_or_else(|| anyhow!("mixed batch must persist its tool child"))?;
    assert_eq!(tool_child.kind, TaskKind::ToolRuntime);
    assert_eq!(tool_child.status, TaskStatus::Pending);
    assert_eq!(tool_child.spawn_index, Some(2));
    assert_eq!(tool_child.thread_id, parent.thread_id);

    // Both kinds are live children of the same parent, and every pending
    // tool call slot is claimed exactly once.
    let children = task_store.list_children(&parent.id).await?;
    assert_eq!(children.len(), 3);
    let mut slots: Vec<u32> = children
        .iter()
        .filter_map(|child| child.spawn_index)
        .collect();
    slots.sort_unstable();
    assert_eq!(slots, vec![0, 1, 2]);

    // Only the subagent half emits SubagentProgress on the parent thread.
    let parent_events = event_repo.get_events(&parent.thread_id).await?;
    assert_eq!(parent_events.len(), 2);
    for event in &parent_events {
        match &event.event {
            AgentEvent::SubagentProgress { completed, .. } => assert!(!completed),
            other => anyhow::bail!("expected SubagentProgress, got {other:?}"),
        }
    }

    Ok(())
}

#[tokio::test]
async fn spawn_mixed_rejects_losing_cas_without_partial_spawn() -> Result<()> {
    // A stale lease loses the CAS: neither the subagent invocation nor
    // the tool child may survive, and the parent stays Running for the
    // worker that still owns it.
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, _lease) = running_parent_root(&task_store).await?;
    let tasks = ["explore A"];
    let payload = mixed_suspension_payload(&parent.thread_id, &tasks, &["todo_write"]);

    let err = spawn_mixed_children_invocations(
        &parent.id,
        &worker,
        &LeaseId::from_string("l-stale"),
        MixedChildrenRequest {
            subagents: mixed_subagent_entries(&tasks)?,
            tool_children: mixed_tool_children(tasks.len(), 2)?,
            payload,
            child_otel_traceparent: None,
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
    .ok_or_else(|| anyhow!("a stale lease must lose the mixed spawn CAS"))?;
    assert!(
        error_text(&err).contains("lease mismatch"),
        "unexpected error: {err:#}"
    );

    let reloaded = task_store
        .get(&parent.id)
        .await?
        .ok_or_else(|| anyhow!("parent must survive a rejected spawn"))?;
    assert_eq!(reloaded.status, TaskStatus::Running);
    assert!(task_store.list_children(&parent.id).await?.is_empty());
    assert!(event_repo.get_events(&parent.thread_id).await?.is_empty());

    Ok(())
}

#[tokio::test]
async fn spawn_mixed_rejects_uncovered_pending_tool_call() -> Result<()> {
    // The batch names slots 0 (subagent) and 1 (tool) but the turn has 3
    // pending tool calls: slot 2 would never resolve and the parent would
    // park on a fan-in it can never finish. Rejected whole, nothing
    // persisted.
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let tasks = ["explore A"];
    let payload = mixed_suspension_payload(&parent.thread_id, &tasks, &["todo_write", "read_file"]);

    let err = spawn_mixed_children_invocations(
        &parent.id,
        &worker,
        &lease,
        MixedChildrenRequest {
            subagents: mixed_subagent_entries(&tasks)?,
            tool_children: mixed_tool_children(tasks.len(), 2)?,
            payload,
            child_otel_traceparent: None,
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
    .ok_or_else(|| anyhow!("an uncovered pending tool call must reject the batch"))?;
    assert!(
        error_text(&err).contains("pending tool calls"),
        "unexpected error: {err:#}"
    );

    let reloaded = task_store
        .get(&parent.id)
        .await?
        .ok_or_else(|| anyhow!("parent must survive a rejected spawn"))?;
    assert_eq!(reloaded.status, TaskStatus::Running);
    assert!(task_store.list_children(&parent.id).await?.is_empty());

    Ok(())
}

#[tokio::test]
async fn spawn_mixed_rejects_bad_tool_slot_without_orphaning_child_threads() -> Result<()> {
    // A tool child pointing at an out-of-range slot must be caught before
    // any child thread is materialized: thread projections are written
    // outside the store's transaction, so a batch that only failed once
    // it reached the store would leave them behind with no task ever
    // referencing them.
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let tasks = ["explore A"];
    // Two pending tool calls: slot 0 (subagent) + slot 1 (tool).
    let payload = mixed_suspension_payload(&parent.thread_id, &tasks, &["todo_write"]);

    let subagents = mixed_subagent_entries(&tasks)?;
    let child_thread_ids: Vec<agent_sdk_foundation::ThreadId> = subagents
        .iter()
        .map(|entry| entry.child_thread_id.clone())
        .collect();
    let err = spawn_mixed_children_invocations(
        &parent.id,
        &worker,
        &lease,
        MixedChildrenRequest {
            subagents,
            tool_children: vec![ToolChildSpawn {
                spawn_index: 7, // out of range — only 2 pending tool calls
                spec: crate::journal::ChildSpawnSpec::new(3),
            }],
            payload,
            child_otel_traceparent: None,
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
    .ok_or_else(|| anyhow!("an out-of-range tool slot must reject the batch"))?;
    assert!(
        error_text(&err).contains("out of bounds"),
        "unexpected error: {err:#}"
    );

    // Nothing was materialized anywhere: no tasks, and no orphan child
    // thread projections.
    let reloaded = task_store
        .get(&parent.id)
        .await?
        .ok_or_else(|| anyhow!("parent must survive a rejected spawn"))?;
    assert_eq!(reloaded.status, TaskStatus::Running);
    assert!(task_store.list_children(&parent.id).await?.is_empty());
    for child_thread_id in &child_thread_ids {
        assert!(
            thread_store.get(child_thread_id).await?.is_none(),
            "a rejected mixed batch must not leave an orphan child thread behind",
        );
    }

    Ok(())
}

#[tokio::test]
async fn spawn_mixed_records_child_ids_in_slot_order_when_interleaved() -> Result<()> {
    // The parent's child-id vector is positional — a later
    // `repark_after_steering` re-derives each child's spawn_index from
    // its position in that vector. An interleaved batch (tool at slot 0,
    // subagent at slot 1) must therefore persist its children ordered by
    // slot, not grouped by kind.
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;

    // Slot 0 → tool call, slot 1 → subagent call.
    let payload = mixed_suspension_payload(&parent.thread_id, &["explore A"], &["todo_write"]);
    let mut payload = payload;
    payload.continuation.payload.pending_tool_calls.swap(0, 1);

    let subagents = vec![SubagentBatchEntry {
        child_thread_id: agent_sdk_foundation::ThreadId::new(),
        spec: sample_spec("explore A"),
        child_root_input: child_root_input("explore A"),
        spawn_index: 1,
        child_caller_metadata: None,
    }];
    let batch = spawn_mixed_children_invocations(
        &parent.id,
        &worker,
        &lease,
        MixedChildrenRequest {
            subagents,
            tool_children: vec![ToolChildSpawn {
                spawn_index: 0,
                spec: crate::journal::ChildSpawnSpec::new(3),
            }],
            payload,
            child_otel_traceparent: None,
        },
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
            event_repo: &event_repo,
        },
        t_plus(2),
    )
    .await?;

    let child_ids = batch.parent_task.state.child_ids();
    assert_eq!(child_ids.len(), 2);
    for (position, child_id) in child_ids.iter().enumerate() {
        let child = task_store
            .get(child_id)
            .await?
            .ok_or_else(|| anyhow!("child {child_id} must be persisted"))?;
        let expected = u32::try_from(position).map_err(|_| anyhow!("position exceeds u32"))?;
        assert_eq!(
            child.spawn_index,
            Some(expected),
            "the child at vector position {position} must resolve slot {position}",
        );
    }

    // Position 0 is the tool child, position 1 the subagent invocation.
    let first = task_store
        .get(&child_ids[0])
        .await?
        .ok_or_else(|| anyhow!("first child"))?;
    assert_eq!(first.kind, TaskKind::ToolRuntime);
    let second = task_store
        .get(&child_ids[1])
        .await?
        .ok_or_else(|| anyhow!("second child"))?;
    assert_eq!(second.kind, TaskKind::Subagent);

    Ok(())
}

#[tokio::test]
async fn spawn_batch_rejects_duplicate_child_thread_without_orphaning_projections() -> Result<()> {
    // Two entries pointing at one child thread is a caller-detectable
    // defect, so it must be caught before any projection is written: the
    // store rejects it too, but only after `get_or_create` has already
    // committed the thread row outside the store's transaction.
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let tasks = vec!["explore A", "explore B"];
    let payload = parent_suspension_payload_with_tools(&parent.thread_id, &tasks);

    let shared_thread = agent_sdk_foundation::ThreadId::new();
    let entries = vec![
        SubagentBatchEntry {
            child_thread_id: shared_thread.clone(),
            spec: sample_spec("explore A"),
            child_root_input: child_root_input("explore A"),
            spawn_index: 0,
            child_caller_metadata: None,
        },
        SubagentBatchEntry {
            child_thread_id: shared_thread.clone(),
            spec: sample_spec("explore B"),
            child_root_input: child_root_input("explore B"),
            spawn_index: 1,
            child_caller_metadata: None,
        },
    ];

    let err = spawn_subagent_batch_invocations(
        &parent.id,
        &worker,
        &lease,
        entries,
        payload,
        &SubagentInvocationDeps {
            task_store: &task_store,
            thread_store: &thread_store,
            event_repo: &event_repo,
        },
        t_plus(2),
    )
    .await
    .err()
    .ok_or_else(|| anyhow!("a duplicate child_thread_id must reject the batch"))?;
    assert!(
        error_text(&err).contains("duplicate child_thread_id"),
        "unexpected error: {err:#}"
    );

    assert!(
        thread_store.get(&shared_thread).await?.is_none(),
        "a rejected batch must not leave a child-thread projection behind",
    );
    assert!(task_store.list_children(&parent.id).await?.is_empty());

    Ok(())
}

#[tokio::test]
async fn spawn_mixed_rejects_duplicate_child_thread_without_orphaning_projections() -> Result<()> {
    // Same contract on the mixed path: the duplicate is caller-detectable,
    // so it is rejected before a projection exists.
    let task_store = InMemoryAgentTaskStore::new();
    let thread_store = InMemoryThreadStore::new();
    let event_repo = InMemoryEventRepository::new();
    let (parent, worker, lease) = running_parent_root(&task_store).await?;
    let tasks = ["explore A", "explore B"];
    let payload = mixed_suspension_payload(&parent.thread_id, &tasks, &["todo_write"]);

    let shared_thread = agent_sdk_foundation::ThreadId::new();
    let subagents = vec![
        SubagentBatchEntry {
            child_thread_id: shared_thread.clone(),
            spec: sample_spec("explore A"),
            child_root_input: child_root_input("explore A"),
            spawn_index: 0,
            child_caller_metadata: None,
        },
        SubagentBatchEntry {
            child_thread_id: shared_thread.clone(),
            spec: sample_spec("explore B"),
            child_root_input: child_root_input("explore B"),
            spawn_index: 1,
            child_caller_metadata: None,
        },
    ];

    let err = spawn_mixed_children_invocations(
        &parent.id,
        &worker,
        &lease,
        MixedChildrenRequest {
            subagents,
            tool_children: mixed_tool_children(tasks.len(), 3)?,
            payload,
            child_otel_traceparent: None,
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
    .ok_or_else(|| anyhow!("a duplicate child_thread_id must reject the batch"))?;
    assert!(
        error_text(&err).contains("duplicate child_thread_id"),
        "unexpected error: {err:#}"
    );

    assert!(
        thread_store.get(&shared_thread).await?.is_none(),
        "a rejected batch must not leave a child-thread projection behind",
    );
    assert!(task_store.list_children(&parent.id).await?.is_empty());

    Ok(())
}
