//! Durable subagent spawn contract and authoritative spec resolution.
//!
//! The public SDK's `SubagentTool` / `SubagentConfig` are local
//! convenience wrappers. They are not strong enough to serve as the
//! server's durable contract because later phases need one typed
//! request shape plus one authoritative resolution path that narrows
//! caller input through inherited parent constraints before any child
//! work is created.
//!
//! Phase 7.1 introduces exactly that boundary:
//!
//! - [`SubagentSpawnRequest`] captures what a caller asked for.
//! - [`InheritedSubagentConstraints`] captures the server-owned bounds
//!   inherited from the parent context.
//! - [`SubagentSpawnPolicy`] provides hook points for model, turns,
//!   timeout, and capability validation / narrowing.
//! - [`resolve_subagent_spec`] produces a single
//!   [`EffectiveSubagentSpec`] that later phases can persist on
//!   invocation tasks or child threads.
//!
//! Phase 7.2 adds [`spawn_subagent_invocation`], the durable creation
//! path for one parent-visible invocation task plus one child thread
//! and its initial child-thread `root_turn` task.
//!
//! Capability selection is deliberately expressed as a server-defined
//! profile plus an optional allowlist that can only narrow that
//! profile. The resolution path may narrow further through inherited
//! parent constraints, but it never widens to arbitrary registry
//! entries.

use std::collections::{BTreeMap, BTreeSet};

use agent_sdk_core::ThreadId;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::journal::{
    AgentTask, AgentTaskId, AgentTaskStore, LeaseId, SubagentInvocationSpawn, SuspensionPayload,
    Thread, ThreadStore, WorkerId,
};

/// Typed durable request to spawn a subagent.
///
/// The request captures caller intent. It is not authoritative on its
/// own: model, turn, timeout, and capability fields must be resolved
/// through [`InheritedSubagentConstraints`] plus a
/// [`SubagentSpawnPolicy`] before the server creates any durable child
/// work.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentSpawnRequest {
    /// Task the subagent should handle.
    pub task: String,
    /// Optional prompt / instruction bundle for the subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Optional model preference requested by the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional turn budget requested by the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    /// Optional timeout budget requested by the caller, in
    /// milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Optional human-friendly nickname shown in progress events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nickname: Option<String>,
    /// Requested capability profile plus optional narrowing allowlist.
    pub capabilities: SubagentCapabilityRequest,
}

impl SubagentSpawnRequest {
    #[must_use]
    pub fn new(task: impl Into<String>, capabilities: SubagentCapabilityRequest) -> Self {
        Self {
            task: task.into(),
            prompt: None,
            model: None,
            max_turns: None,
            timeout_ms: None,
            nickname: None,
            capabilities,
        }
    }

    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    #[must_use]
    pub const fn with_max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    #[must_use]
    pub fn with_nickname(mut self, nickname: impl Into<String>) -> Self {
        self.nickname = Some(nickname.into());
        self
    }
}

/// Requested capability selection for a subagent spawn.
///
/// The server owns the profile catalog. The caller can choose one
/// server-defined profile and optionally narrow it with an allowlist,
/// but cannot widen beyond that profile or the inherited parent
/// ceiling.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentCapabilityRequest {
    /// Name of the server-defined capability profile.
    pub profile: String,
    /// Optional allowlist that narrows the selected profile.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowlist: BTreeSet<String>,
}

impl SubagentCapabilityRequest {
    #[must_use]
    pub fn new(profile: impl Into<String>) -> Self {
        Self {
            profile: profile.into(),
            allowlist: BTreeSet::new(),
        }
    }

    #[must_use]
    pub fn with_allowlist<I, S>(mut self, allowlist: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowlist = allowlist.into_iter().map(Into::into).collect();
        self
    }
}

/// Server-defined capability profile.
///
/// The profile ID is the key in
/// [`InheritedSubagentConstraints::capability_profiles`]. Storing only
/// the capability set here keeps the durable wire format compact while
/// preserving deterministic ordering via [`BTreeSet`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentCapabilityProfile {
    /// Capability identifiers exposed by the profile.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub capabilities: BTreeSet<String>,
}

/// Parent-bounded constraints inherited by a durable subagent spawn.
///
/// These values are authoritative: a child request cannot exceed them.
/// The capability ceiling represents the parent's already-resolved
/// effective access. Later nested spawns can only narrow from there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InheritedSubagentConstraints {
    /// Default model to use when the request omits one or asks for a
    /// disallowed model.
    pub default_model: String,
    /// Models the parent policy allows this subagent to use.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_models: BTreeSet<String>,
    /// Default turn budget applied when the request omits one.
    pub default_max_turns: u32,
    /// Maximum turn budget inherited from the parent.
    pub max_turns: u32,
    /// Default timeout applied when the request omits one.
    pub default_timeout_ms: u64,
    /// Maximum timeout inherited from the parent.
    pub max_timeout_ms: u64,
    /// Server-defined capability profiles visible to this parent.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub capability_profiles: BTreeMap<String, SubagentCapabilityProfile>,
    /// Effective capability ceiling inherited from the parent.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_capabilities: BTreeSet<String>,
}

impl InheritedSubagentConstraints {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.default_model.trim().is_empty(),
            "default_model cannot be blank",
        );
        ensure!(
            !self.allowed_models.is_empty(),
            "allowed_models cannot be empty",
        );
        ensure!(
            self.allowed_models.contains(&self.default_model),
            "default_model `{}` must be included in allowed_models",
            self.default_model,
        );
        ensure!(
            self.default_max_turns > 0,
            "default_max_turns must be greater than zero",
        );
        ensure!(self.max_turns > 0, "max_turns must be greater than zero");
        ensure!(
            self.default_max_turns <= self.max_turns,
            "default_max_turns ({}) cannot exceed max_turns ({})",
            self.default_max_turns,
            self.max_turns,
        );
        ensure!(
            self.default_timeout_ms > 0,
            "default_timeout_ms must be greater than zero",
        );
        ensure!(
            self.max_timeout_ms > 0,
            "max_timeout_ms must be greater than zero",
        );
        ensure!(
            self.default_timeout_ms <= self.max_timeout_ms,
            "default_timeout_ms ({}) cannot exceed max_timeout_ms ({})",
            self.default_timeout_ms,
            self.max_timeout_ms,
        );
        ensure!(
            !self.capability_profiles.is_empty(),
            "capability_profiles cannot be empty",
        );
        ensure!(
            !self.allowed_capabilities.is_empty(),
            "allowed_capabilities cannot be empty",
        );

        for (profile, definition) in &self.capability_profiles {
            ensure!(
                !profile.trim().is_empty(),
                "capability profile names cannot be blank",
            );
            ensure!(
                !definition.capabilities.is_empty(),
                "capability profile `{profile}` cannot be empty",
            );
            for cap in &definition.capabilities {
                ensure!(
                    !cap.trim().is_empty(),
                    "capability identifier in profile `{profile}` cannot be blank",
                );
            }
        }

        for cap in &self.allowed_capabilities {
            ensure!(
                !cap.trim().is_empty(),
                "allowed_capabilities contains a blank identifier",
            );
        }

        let known_capabilities = self.known_capabilities();
        let unknown_allowed: Vec<_> = self
            .allowed_capabilities
            .difference(&known_capabilities)
            .cloned()
            .collect();
        ensure!(
            unknown_allowed.is_empty(),
            "allowed_capabilities contains unknown entries: {}",
            unknown_allowed.join(", "),
        );

        Ok(())
    }

    fn known_capabilities(&self) -> BTreeSet<String> {
        self.capability_profiles
            .values()
            .flat_map(|profile| profile.capabilities.iter().cloned())
            .collect()
    }
}

/// Effective capability selection after server policy resolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveSubagentCapabilities {
    /// Resolved server-defined profile name.
    pub profile: String,
    /// Effective capabilities after request narrowing plus inherited
    /// parent ceiling enforcement.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed: BTreeSet<String>,
}

/// Server-authoritative durable subagent spec.
///
/// This is the resolved contract later phases should persist on
/// invocation tasks or child threads. It is the only spec the server
/// should treat as authoritative when creating durable subagent work.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveSubagentSpec {
    /// Task the subagent should perform.
    pub task: String,
    /// Resolved prompt bundle. Empty string means no extra prompt.
    pub prompt: String,
    /// Server-authoritative model identifier.
    pub model: String,
    /// Server-authoritative turn budget.
    pub max_turns: u32,
    /// Server-authoritative timeout, in milliseconds.
    pub timeout_ms: u64,
    /// Optional human-friendly nickname for progress reporting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nickname: Option<String>,
    /// Effective capability selection.
    pub capabilities: EffectiveSubagentCapabilities,
}

/// Policy hook surface for subagent spawn resolution.
///
/// The default server implementation clamps numeric budgets and
/// intersects capability requests with the inherited ceiling, but
/// deployments can inject a stricter policy if needed.
pub trait SubagentSpawnPolicy: Send + Sync {
    /// Resolve the authoritative model identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if policy infrastructure cannot determine a
    /// valid model for the given request and inherited constraints.
    fn resolve_model(
        &self,
        requested: Option<&str>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<String>;

    /// Resolve the authoritative turn budget.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy cannot produce a valid positive
    /// turn budget for the given request and inherited constraints.
    fn resolve_max_turns(
        &self,
        requested: Option<u32>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<u32>;

    /// Resolve the authoritative timeout budget.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy cannot produce a valid positive
    /// timeout for the given request and inherited constraints.
    fn resolve_timeout_ms(
        &self,
        requested: Option<u64>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<u64>;

    /// Resolve the authoritative capability selection.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested profile is unknown or if the
    /// resulting capability set would violate the inherited
    /// constraints.
    fn resolve_capabilities(
        &self,
        requested: &SubagentCapabilityRequest,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentCapabilities>;
}

/// Default server-side subagent spawn policy.
///
/// - Models fall back to the inherited default when the request asks
///   for a disallowed one.
/// - Turn and timeout budgets are clamped to the inherited maximum.
/// - Capability selections must name a known profile and can only
///   narrow it. The inherited parent ceiling may narrow further.
#[derive(Clone, Debug, Default)]
pub struct ServerSubagentSpawnPolicy;

impl SubagentSpawnPolicy for ServerSubagentSpawnPolicy {
    fn resolve_model(
        &self,
        requested: Option<&str>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<String> {
        let requested_model = requested.unwrap_or(constraints.default_model.as_str());
        if constraints.allowed_models.contains(requested_model) {
            Ok(requested_model.to_owned())
        } else {
            Ok(constraints.default_model.clone())
        }
    }

    fn resolve_max_turns(
        &self,
        requested: Option<u32>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<u32> {
        let requested = requested.unwrap_or(constraints.default_max_turns);
        ensure!(
            requested > 0,
            "requested max_turns must be greater than zero"
        );
        Ok(requested.min(constraints.max_turns))
    }

    fn resolve_timeout_ms(
        &self,
        requested: Option<u64>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<u64> {
        let requested = requested.unwrap_or(constraints.default_timeout_ms);
        ensure!(
            requested > 0,
            "requested timeout_ms must be greater than zero",
        );
        Ok(requested.min(constraints.max_timeout_ms))
    }

    fn resolve_capabilities(
        &self,
        requested: &SubagentCapabilityRequest,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentCapabilities> {
        let profile_name = requested.profile.trim();
        let profile = constraints
            .capability_profiles
            .get(profile_name)
            .with_context(|| format!("unknown capability profile `{profile_name}`"))?;

        let invalid_allowlist: Vec<_> = requested
            .allowlist
            .difference(&profile.capabilities)
            .cloned()
            .collect();
        ensure!(
            invalid_allowlist.is_empty(),
            "capability allowlist can only narrow profile `{}`; unsupported entries: {}",
            profile_name,
            invalid_allowlist.join(", "),
        );

        let requested_capabilities = if requested.allowlist.is_empty() {
            profile.capabilities.clone()
        } else {
            requested.allowlist.clone()
        };

        let allowed = requested_capabilities
            .intersection(&constraints.allowed_capabilities)
            .cloned()
            .collect::<BTreeSet<_>>();

        ensure!(
            !allowed.is_empty(),
            "capability selection for profile `{profile_name}` resolves to no allowed capabilities",
        );

        Ok(EffectiveSubagentCapabilities {
            profile: profile_name.to_owned(),
            allowed,
        })
    }
}

/// Resolve a typed spawn request into one authoritative effective
/// subagent spec.
///
/// The resolution path is deterministic for a given `(request,
/// constraints, policy)` triple. This is the boundary later durable
/// subagent phases should call before they create child threads or
/// invocation tasks.
///
/// # Errors
///
/// Returns an error if:
/// - the request is structurally invalid,
/// - the inherited constraints are internally inconsistent, or
/// - the supplied policy cannot resolve one of the authoritative
///   fields.
pub fn resolve_subagent_spec(
    request: &SubagentSpawnRequest,
    constraints: &InheritedSubagentConstraints,
    policy: &dyn SubagentSpawnPolicy,
) -> Result<EffectiveSubagentSpec> {
    validate_request(request)?;
    constraints
        .validate()
        .context("invalid inherited subagent constraints")?;

    let model = policy
        .resolve_model(request.model.as_deref(), constraints)
        .context("resolve subagent model")?;
    let max_turns = policy
        .resolve_max_turns(request.max_turns, constraints)
        .context("resolve subagent max_turns")?;
    let timeout_ms = policy
        .resolve_timeout_ms(request.timeout_ms, constraints)
        .context("resolve subagent timeout_ms")?;
    let capabilities = policy
        .resolve_capabilities(&request.capabilities, constraints)
        .context("resolve subagent capabilities")?;

    Ok(EffectiveSubagentSpec {
        task: request.task.trim().to_owned(),
        prompt: normalize_optional_string(request.prompt.as_deref()).unwrap_or_default(),
        model,
        max_turns,
        timeout_ms,
        nickname: normalize_optional_string(request.nickname.as_deref()),
        capabilities,
    })
}

/// Durable stores needed to persist a subagent invocation and
/// materialize its child thread projection.
pub struct SubagentInvocationDeps<'a> {
    pub task_store: &'a dyn AgentTaskStore,
    pub thread_store: &'a dyn ThreadStore,
}

/// Durable records created for one subagent spawn.
pub struct SpawnedSubagentInvocation {
    pub parent_task: AgentTask,
    pub invocation_task: AgentTask,
    pub child_thread: Thread,
    pub child_root_task: AgentTask,
}

/// Persist one durable subagent invocation under a running parent.
///
/// This is the Phase 7.2 creation path:
///
/// - parent root task parks in `waiting_on_children`,
/// - one parent-visible `subagent` invocation task is created,
/// - one child thread is allocated and materialized before any child
///   task row is persisted,
/// - one initial child-thread `root_turn` task is admitted as
///   runnable,
/// - durable linkage among those records is persisted on the
///   invocation task.
///
/// # Errors
///
/// Returns an error if the parent task cannot be parked on child
/// execution, if the invocation and child root tasks cannot be
/// persisted, if the durable linkage is inconsistent, or if the child
/// thread projection cannot be materialized.
pub async fn spawn_subagent_invocation(
    parent_id: &AgentTaskId,
    worker: &WorkerId,
    lease: &LeaseId,
    spec: EffectiveSubagentSpec,
    payload: SuspensionPayload,
    deps: &SubagentInvocationDeps<'_>,
    now: OffsetDateTime,
) -> Result<SpawnedSubagentInvocation> {
    let child_thread_id = ThreadId::new();
    let child_thread = deps
        .thread_store
        .get_or_create(&child_thread_id, now)
        .await
        .context("materialize child thread projection")?;

    let (parent_task, invocation_task, child_root_task) = deps
        .task_store
        .spawn_subagent_invocation(
            parent_id,
            worker,
            lease,
            SubagentInvocationSpawn {
                child_thread_id,
                spec,
                payload,
            },
            now,
        )
        .await
        .context("persist subagent invocation tasks")?;

    let linkage = invocation_task
        .state
        .subagent_invocation()
        .context("subagent invocation task missing durable linkage")?;
    ensure!(
        linkage.child_root_task_id == child_root_task.id,
        "subagent invocation linkage points at child root {} but store returned {}",
        linkage.child_root_task_id,
        child_root_task.id,
    );
    ensure!(
        linkage.child_thread_id == child_root_task.thread_id,
        "subagent invocation linkage points at child thread {} but child root uses {}",
        linkage.child_thread_id,
        child_root_task.thread_id,
    );

    ensure!(
        child_thread.thread_id == child_root_task.thread_id,
        "materialized child thread {} but child root uses {}",
        child_thread.thread_id,
        child_root_task.thread_id,
    );

    Ok(SpawnedSubagentInvocation {
        parent_task,
        invocation_task,
        child_thread,
        child_root_task,
    })
}

fn validate_request(request: &SubagentSpawnRequest) -> Result<()> {
    ensure!(!request.task.trim().is_empty(), "task cannot be blank");
    ensure!(
        !request.capabilities.profile.trim().is_empty(),
        "capability profile cannot be blank",
    );
    if let Some(ref model) = request.model {
        ensure!(!model.trim().is_empty(), "model cannot be blank");
    }
    if let Some(max_turns) = request.max_turns {
        ensure!(max_turns > 0, "max_turns must be greater than zero");
    }
    if let Some(timeout_ms) = request.timeout_ms {
        ensure!(timeout_ms > 0, "timeout_ms must be greater than zero");
    }
    Ok(())
}

fn normalize_optional_string(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}
