//! Durable subagent spawn contract tests.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow};

use super::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentSpec, InheritedSubagentConstraints,
    ServerSubagentSpawnPolicy, SubagentCapabilityProfile, SubagentCapabilityRequest,
    SubagentSpawnPolicy, SubagentSpawnRequest, resolve_subagent_spec,
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
