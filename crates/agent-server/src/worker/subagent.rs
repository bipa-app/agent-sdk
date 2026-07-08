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

use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ThreadId, TokenUsage, ToolResult, ToolTier, llm};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::journal::message_store::MessageProjectionStore;
use crate::journal::task::SubmittedInputItem;
use crate::journal::{
    AgentTask, AgentTaskId, AgentTaskStore, CommittedEvent, EventRepository, LeaseId,
    SubagentInvocationSpawn, SuspensionPayload, TaskKind, TaskStatus, Thread, ThreadStore,
    WorkerId,
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
    /// Optional nested-subagent sibling budget requested by the
    /// caller. Zero forbids nested subagent spawns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_parallel_subagents: Option<u32>,
    /// Optional human-friendly nickname shown in progress events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nickname: Option<String>,
    /// Optional sandbox narrowing for this subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SubagentSandboxPolicy>,
    /// Optional MCP-server narrowing for this subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<SubagentMcpRequest>,
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
            max_parallel_subagents: None,
            nickname: None,
            sandbox: None,
            mcp: None,
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
    pub const fn with_max_parallel_subagents(mut self, max_parallel_subagents: u32) -> Self {
        self.max_parallel_subagents = Some(max_parallel_subagents);
        self
    }

    #[must_use]
    pub fn with_nickname(mut self, nickname: impl Into<String>) -> Self {
        self.nickname = Some(nickname.into());
        self
    }

    #[must_use]
    pub const fn with_sandbox(mut self, sandbox: SubagentSandboxPolicy) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    #[must_use]
    pub fn with_mcp_allowlist<I, S>(mut self, allowlist: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.mcp = Some(SubagentMcpRequest::with_allowlist(allowlist));
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

/// Ordered sandbox modes for durable subagent policy narrowing.
///
/// The declaration order is from the narrowest to the widest mode so
/// `Ord` comparisons line up with authority widening.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentSandboxMode {
    #[default]
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}

/// Sandbox policy carried through durable subagent specs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentSandboxPolicy {
    /// Filesystem / process authority level.
    #[serde(default)]
    pub mode: SubagentSandboxMode,
    /// Whether outbound network access is allowed.
    #[serde(default)]
    pub network_access: bool,
}

impl SubagentSandboxPolicy {
    #[must_use]
    pub const fn read_only() -> Self {
        Self {
            mode: SubagentSandboxMode::ReadOnly,
            network_access: false,
        }
    }

    #[must_use]
    pub const fn workspace_write() -> Self {
        Self {
            mode: SubagentSandboxMode::WorkspaceWrite,
            network_access: false,
        }
    }

    #[must_use]
    pub const fn full_access() -> Self {
        Self {
            mode: SubagentSandboxMode::FullAccess,
            network_access: true,
        }
    }

    #[must_use]
    pub const fn with_network_access(mut self, network_access: bool) -> Self {
        self.network_access = network_access;
        self
    }

    #[must_use]
    pub fn narrow_to(&self, other: &Self) -> Self {
        Self {
            mode: self.mode.min(other.mode),
            network_access: self.network_access && other.network_access,
        }
    }
}

/// Optional MCP-server allowlist narrowing requested by the caller.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentMcpRequest {
    /// Optional allowlist. `None` means "inherit the profile default";
    /// `Some(empty)` means "disable MCP access".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowlist: Option<BTreeSet<String>>,
}

impl SubagentMcpRequest {
    #[must_use]
    pub fn with_allowlist<I, S>(allowlist: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowlist: Some(allowlist.into_iter().map(Into::into).collect()),
        }
    }
}

/// Server-defined capability profile.
///
/// The profile ID is the key in
/// [`InheritedSubagentPolicy::capability_profiles`]. Storing only
/// the capability set here keeps the durable wire format compact while
/// preserving deterministic ordering via [`BTreeSet`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentCapabilityProfile {
    /// Capability identifiers exposed by the profile.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub capabilities: BTreeSet<String>,
    /// Sandbox ceiling associated with this profile.
    #[serde(default)]
    pub sandbox: SubagentSandboxPolicy,
    /// MCP servers visible through this profile.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_mcp_servers: BTreeSet<String>,
}

/// Durable inherited policy ceiling carried forward for nested subagents.
///
/// This is the reusable policy object that later nested subagent
/// spawns inherit. Runtime counters such as the current depth and
/// current active-sibling count live outside this struct so they can be
/// recomputed from durable task state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InheritedSubagentPolicy {
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
    /// Maximum nested subagent depth this lineage may reach.
    pub max_depth: u32,
    /// Maximum number of live subagent siblings this lineage may have.
    pub max_parallel_subagents: u32,
    /// Inherited sandbox ceiling.
    #[serde(default)]
    pub sandbox: SubagentSandboxPolicy,
    /// Inherited MCP-server ceiling.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_mcp_servers: BTreeSet<String>,
    /// Inherited provider / audit context.
    pub audit_provider: String,
}

impl InheritedSubagentPolicy {
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
        ensure!(
            !self.audit_provider.trim().is_empty(),
            "audit_provider cannot be blank",
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
            for server in &definition.allowed_mcp_servers {
                ensure!(
                    !server.trim().is_empty(),
                    "MCP server identifier in profile `{profile}` cannot be blank",
                );
            }
        }

        for cap in &self.allowed_capabilities {
            ensure!(
                !cap.trim().is_empty(),
                "allowed_capabilities contains a blank identifier",
            );
        }

        for server in &self.allowed_mcp_servers {
            ensure!(
                !server.trim().is_empty(),
                "allowed_mcp_servers contains a blank identifier",
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

/// Parent-bounded constraints inherited by a durable subagent spawn.
///
/// These values are authoritative: a child request cannot exceed them.
/// The capability ceiling represents the parent's already-resolved
/// effective access. Later nested spawns can only narrow from there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InheritedSubagentConstraints {
    /// Reusable inherited policy ceiling.
    pub policy: InheritedSubagentPolicy,
    /// Current durable depth of the would-be parent.
    pub current_depth: u32,
    /// Current number of live subagent siblings under the same parent.
    pub active_parallel_subagents: u32,
}

impl InheritedSubagentConstraints {
    fn validate(&self) -> Result<()> {
        self.policy
            .validate()
            .context("invalid inherited subagent policy")?;
        ensure!(
            self.current_depth <= self.policy.max_depth,
            "current_depth ({}) cannot exceed max_depth ({})",
            self.current_depth,
            self.policy.max_depth,
        );
        ensure!(
            self.active_parallel_subagents <= self.policy.max_parallel_subagents,
            "active_parallel_subagents ({}) cannot exceed max_parallel_subagents ({})",
            self.active_parallel_subagents,
            self.policy.max_parallel_subagents,
        );

        Ok(())
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

/// Effective MCP-server policy after server resolution.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveSubagentMcpPolicy {
    /// Effective MCP servers visible to the subagent.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_servers: BTreeSet<String>,
}

/// Server-authoritative durable subagent spec.
///
/// This is the resolved contract later phases should persist on
/// invocation tasks or child threads. It is the only spec the server
/// should treat as authoritative when creating durable subagent work.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "EffectiveSubagentSpecWire")]
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
    /// Durable depth of this child within the subagent lineage.
    #[serde(default = "default_subagent_depth")]
    pub depth: u32,
    /// Maximum live nested-subagent siblings this child may create.
    #[serde(default)]
    pub max_parallel_subagents: u32,
    /// Optional human-friendly nickname for progress reporting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nickname: Option<String>,
    /// Effective sandbox policy for this subagent.
    #[serde(default)]
    pub sandbox: SubagentSandboxPolicy,
    /// Effective MCP policy for this subagent.
    #[serde(default)]
    pub mcp: EffectiveSubagentMcpPolicy,
    /// Audit provenance enforced by the server for this subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_provenance: Option<AuditProvenance>,
    /// Durable inherited policy ceiling for later nested spawns.
    pub inherited_policy: InheritedSubagentPolicy,
    /// Effective capability selection.
    pub capabilities: EffectiveSubagentCapabilities,
}

#[derive(Deserialize)]
struct EffectiveSubagentSpecWire {
    task: String,
    prompt: String,
    model: String,
    max_turns: u32,
    timeout_ms: u64,
    #[serde(default = "default_subagent_depth")]
    depth: u32,
    #[serde(default)]
    max_parallel_subagents: u32,
    #[serde(default)]
    nickname: Option<String>,
    #[serde(default)]
    sandbox: SubagentSandboxPolicy,
    #[serde(default)]
    mcp: EffectiveSubagentMcpPolicy,
    #[serde(default)]
    audit_provenance: Option<AuditProvenance>,
    #[serde(default)]
    inherited_policy: Option<InheritedSubagentPolicy>,
    capabilities: EffectiveSubagentCapabilities,
}

impl From<EffectiveSubagentSpecWire> for EffectiveSubagentSpec {
    fn from(wire: EffectiveSubagentSpecWire) -> Self {
        let inherited_policy = wire
            .inherited_policy
            .clone()
            .unwrap_or_else(|| wire.legacy_inherited_policy());
        let EffectiveSubagentSpecWire {
            task,
            prompt,
            model,
            max_turns,
            timeout_ms,
            depth,
            max_parallel_subagents,
            nickname,
            sandbox,
            mcp,
            audit_provenance,
            inherited_policy: _,
            capabilities,
        } = wire;

        Self {
            task,
            prompt,
            model,
            max_turns,
            timeout_ms,
            depth,
            max_parallel_subagents,
            nickname,
            sandbox,
            mcp,
            audit_provenance,
            inherited_policy,
            capabilities,
        }
    }
}

impl EffectiveSubagentSpecWire {
    fn legacy_inherited_policy(&self) -> InheritedSubagentPolicy {
        let default_model =
            normalize_optional_string(Some(&self.model)).unwrap_or_else(|| "unknown".into());
        let allowed_capabilities = if self.capabilities.allowed.is_empty() {
            BTreeSet::from([legacy_unavailable_capability().to_owned()])
        } else {
            self.capabilities.allowed.clone()
        };
        let profile_name = normalize_optional_string(Some(&self.capabilities.profile))
            .unwrap_or_else(|| legacy_effective_profile_name().to_owned());
        let audit_provider = self
            .audit_provenance
            .as_ref()
            .and_then(|audit| normalize_optional_string(Some(&audit.provider)))
            .unwrap_or_else(|| "unknown".into());
        let mut capability_profiles = BTreeMap::new();
        capability_profiles.insert(
            profile_name,
            SubagentCapabilityProfile {
                capabilities: allowed_capabilities.clone(),
                sandbox: self.sandbox.clone(),
                allowed_mcp_servers: self.mcp.allowed_servers.clone(),
            },
        );

        InheritedSubagentPolicy {
            default_model: default_model.clone(),
            allowed_models: BTreeSet::from([default_model]),
            default_max_turns: self.max_turns.max(1),
            max_turns: self.max_turns.max(1),
            default_timeout_ms: self.timeout_ms.max(1),
            max_timeout_ms: self.timeout_ms.max(1),
            capability_profiles,
            allowed_capabilities,
            max_depth: self.depth.max(default_subagent_depth()),
            max_parallel_subagents: self.max_parallel_subagents.max(1),
            sandbox: self.sandbox.clone(),
            allowed_mcp_servers: self.mcp.allowed_servers.clone(),
            audit_provider,
        }
    }
}

impl EffectiveSubagentSpec {
    /// Build the runtime constraints a nested child would inherit from
    /// this already-resolved spec.
    #[must_use]
    pub fn inherited_constraints(
        &self,
        active_parallel_subagents: u32,
    ) -> InheritedSubagentConstraints {
        InheritedSubagentConstraints {
            policy: self.inherited_policy.clone(),
            current_depth: self.depth,
            active_parallel_subagents,
        }
    }
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

    /// Resolve the authoritative nested-subagent sibling budget.
    ///
    /// # Errors
    ///
    /// Returns an error if the current parent has already exhausted its
    /// inherited parallel-subagent budget.
    fn resolve_max_parallel_subagents(
        &self,
        requested: Option<u32>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<u32>;

    /// Resolve the authoritative child depth.
    ///
    /// # Errors
    ///
    /// Returns an error if the inherited depth ceiling is already
    /// exhausted.
    fn resolve_depth(&self, constraints: &InheritedSubagentConstraints) -> Result<u32>;

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

    /// Resolve the authoritative sandbox policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the capability profile is unknown.
    fn resolve_sandbox(
        &self,
        profile: &str,
        requested: Option<&SubagentSandboxPolicy>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<SubagentSandboxPolicy>;

    /// Resolve the authoritative MCP policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the capability profile is unknown or if the
    /// request names MCP servers outside that profile.
    fn resolve_mcp(
        &self,
        profile: &str,
        requested: Option<&SubagentMcpRequest>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentMcpPolicy>;

    /// Resolve the authoritative audit provenance.
    ///
    /// # Errors
    ///
    /// Returns an error if the deployment cannot construct a trusted
    /// audit context for the resolved child model.
    fn resolve_audit_provenance(
        &self,
        model: &str,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<AuditProvenance>;
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
        let requested_model = requested.unwrap_or(constraints.policy.default_model.as_str());
        if constraints.policy.allowed_models.contains(requested_model) {
            Ok(requested_model.to_owned())
        } else {
            // Loud fallback so the substitution is searchable in
            // operational logs — the caller's requested model is
            // otherwise only recoverable by diffing request.model
            // against the resolved spec.model after the fact.
            log::warn!(
                "subagent requested disallowed model {requested_model:?}; \
                 substituting inherited default {:?}",
                constraints.policy.default_model,
            );
            Ok(constraints.policy.default_model.clone())
        }
    }

    fn resolve_max_turns(
        &self,
        requested: Option<u32>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<u32> {
        let requested = requested.unwrap_or(constraints.policy.default_max_turns);
        ensure!(
            requested > 0,
            "requested max_turns must be greater than zero"
        );
        Ok(requested.min(constraints.policy.max_turns))
    }

    fn resolve_timeout_ms(
        &self,
        requested: Option<u64>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<u64> {
        let requested = requested.unwrap_or(constraints.policy.default_timeout_ms);
        ensure!(
            requested > 0,
            "requested timeout_ms must be greater than zero",
        );
        Ok(requested.min(constraints.policy.max_timeout_ms))
    }

    fn resolve_max_parallel_subagents(
        &self,
        requested: Option<u32>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<u32> {
        ensure!(
            constraints.active_parallel_subagents < constraints.policy.max_parallel_subagents,
            "parallel subagent budget exhausted ({}/{})",
            constraints.active_parallel_subagents,
            constraints.policy.max_parallel_subagents,
        );
        Ok(requested
            .unwrap_or(constraints.policy.max_parallel_subagents)
            .min(constraints.policy.max_parallel_subagents))
    }

    fn resolve_depth(&self, constraints: &InheritedSubagentConstraints) -> Result<u32> {
        ensure!(
            constraints.current_depth < constraints.policy.max_depth,
            "subagent depth limit exceeded ({}/{})",
            constraints.current_depth,
            constraints.policy.max_depth,
        );
        Ok(constraints.current_depth.saturating_add(1))
    }

    fn resolve_capabilities(
        &self,
        requested: &SubagentCapabilityRequest,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentCapabilities> {
        let profile_name = requested.profile.trim();
        let profile = constraints
            .policy
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
            .intersection(&constraints.policy.allowed_capabilities)
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

    fn resolve_sandbox(
        &self,
        profile: &str,
        requested: Option<&SubagentSandboxPolicy>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<SubagentSandboxPolicy> {
        let profile = constraints
            .policy
            .capability_profiles
            .get(profile)
            .with_context(|| format!("unknown capability profile `{profile}`"))?;
        let requested = requested.unwrap_or(&profile.sandbox);
        Ok(constraints
            .policy
            .sandbox
            .narrow_to(&profile.sandbox)
            .narrow_to(requested))
    }

    fn resolve_mcp(
        &self,
        profile: &str,
        requested: Option<&SubagentMcpRequest>,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<EffectiveSubagentMcpPolicy> {
        let profile = constraints
            .policy
            .capability_profiles
            .get(profile)
            .with_context(|| format!("unknown capability profile `{profile}`"))?;
        let requested_servers =
            if let Some(allowlist) = requested.and_then(|requested| requested.allowlist.as_ref()) {
                let invalid_allowlist: Vec<_> = allowlist
                    .difference(&profile.allowed_mcp_servers)
                    .cloned()
                    .collect();
                ensure!(
                    invalid_allowlist.is_empty(),
                    "MCP allowlist can only narrow profile servers; unsupported entries: {}",
                    invalid_allowlist.join(", "),
                );
                allowlist.clone()
            } else {
                profile.allowed_mcp_servers.clone()
            };

        let allowed_servers = requested_servers
            .intersection(&constraints.policy.allowed_mcp_servers)
            .cloned()
            .collect();
        Ok(EffectiveSubagentMcpPolicy { allowed_servers })
    }

    fn resolve_audit_provenance(
        &self,
        model: &str,
        constraints: &InheritedSubagentConstraints,
    ) -> Result<AuditProvenance> {
        Ok(AuditProvenance::new(
            constraints.policy.audit_provider.clone(),
            model.to_owned(),
        ))
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
    let max_parallel_subagents = policy
        .resolve_max_parallel_subagents(request.max_parallel_subagents, constraints)
        .context("resolve subagent max_parallel_subagents")?;
    let depth = policy
        .resolve_depth(constraints)
        .context("resolve subagent depth")?;
    let capabilities = policy
        .resolve_capabilities(&request.capabilities, constraints)
        .context("resolve subagent capabilities")?;
    let profile_name = capabilities.profile.as_str();
    let sandbox = policy
        .resolve_sandbox(profile_name, request.sandbox.as_ref(), constraints)
        .context("resolve subagent sandbox")?;
    let mcp = policy
        .resolve_mcp(profile_name, request.mcp.as_ref(), constraints)
        .context("resolve subagent MCP policy")?;
    let audit_provenance = policy
        .resolve_audit_provenance(&model, constraints)
        .context("resolve subagent audit provenance")?;
    let inherited_model = model.clone();
    let inherited_allowed_mcp_servers = mcp.allowed_servers.clone();

    Ok(EffectiveSubagentSpec {
        task: request.task.trim().to_owned(),
        prompt: normalize_optional_string(request.prompt.as_deref()).unwrap_or_default(),
        model,
        max_turns,
        timeout_ms,
        depth,
        max_parallel_subagents,
        nickname: normalize_optional_string(request.nickname.as_deref()),
        sandbox: sandbox.clone(),
        mcp,
        audit_provenance: Some(audit_provenance),
        inherited_policy: InheritedSubagentPolicy {
            default_model: inherited_model.clone(),
            allowed_models: BTreeSet::from([inherited_model]),
            default_max_turns: max_turns,
            max_turns,
            default_timeout_ms: timeout_ms,
            max_timeout_ms: timeout_ms,
            capability_profiles: constraints.policy.capability_profiles.clone(),
            allowed_capabilities: capabilities.allowed.clone(),
            max_depth: constraints.policy.max_depth,
            max_parallel_subagents,
            sandbox,
            allowed_mcp_servers: inherited_allowed_mcp_servers,
            audit_provider: constraints.policy.audit_provider.clone(),
        },
        capabilities,
    })
}

/// Durable stores needed to persist a subagent invocation and
/// materialize its child thread projection.
pub struct SubagentInvocationDeps<'a> {
    pub task_store: &'a dyn AgentTaskStore,
    pub thread_store: &'a dyn ThreadStore,
    pub event_repo: &'a dyn EventRepository,
}

/// Durable records created for one subagent spawn.
#[derive(Debug)]
pub struct SpawnedSubagentInvocation {
    pub parent_task: AgentTask,
    pub invocation_task: AgentTask,
    pub child_thread: Thread,
    pub child_root_task: AgentTask,
    pub committed_events: Vec<CommittedEvent>,
}

/// Structured summary materialized from a terminal child thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentSummary {
    pub success: bool,
    pub total_turns: u32,
    pub tool_count: u32,
    pub total_usage: TokenUsage,
    pub duration_ms: u64,
}

/// Final durable subagent result returned to the parent-facing tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentResult {
    pub final_response: String,
    pub summary: SubagentSummary,
    pub child_thread_id: ThreadId,
    pub child_root_task_id: AgentTaskId,
    pub subagent_task_id: AgentTaskId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_details: Option<String>,
}

/// Validated bootstrap for a running `subagent` invocation task.
#[derive(Clone, Debug)]
pub struct SubagentTaskBootstrap {
    pub invocation_task: AgentTask,
    pub thread_id: ThreadId,
    pub task_id: AgentTaskId,
    pub worker_id: WorkerId,
    pub lease_id: LeaseId,
    pub subagent_id: String,
    pub subagent_name: String,
    pub child_thread_id: ThreadId,
    pub child_root_task_id: AgentTaskId,
    pub spec: EffectiveSubagentSpec,
}

/// Durable stores needed to materialize a subagent result from a
/// terminal child thread.
pub struct SubagentResultDeps<'a> {
    pub task_store: &'a dyn AgentTaskStore,
    pub thread_store: &'a dyn ThreadStore,
    pub message_store: &'a dyn MessageProjectionStore,
    pub event_repo: &'a dyn EventRepository,
}

/// Successful completion of a `subagent` invocation task.
pub struct SubagentTaskOutcome {
    pub invocation_task: AgentTask,
    pub parent_task: Option<AgentTask>,
    pub subagent_result: SubagentResult,
    pub tool_result: ToolResult,
    pub committed_events: Vec<CommittedEvent>,
}

/// Resolve a running `subagent` invocation task into trusted
/// materialization inputs.
///
/// # Errors
///
/// Returns an error if the task is not a running `subagent`
/// invocation, if the durable invocation linkage is missing, or if
/// the parent task cannot be loaded.
pub async fn resolve_subagent_bootstrap(
    task: AgentTask,
    task_store: &dyn AgentTaskStore,
) -> Result<SubagentTaskBootstrap> {
    ensure!(
        task.kind == TaskKind::Subagent,
        "subagent bootstrap requires a Subagent task, got {:?}",
        task.kind,
    );
    ensure!(
        task.status == TaskStatus::Running,
        "subagent bootstrap requires a Running task, got {:?}",
        task.status,
    );

    let worker_id = task
        .worker_id
        .clone()
        .context("running subagent task missing worker_id")?;
    let lease_id = task
        .lease_id
        .clone()
        .context("running subagent task missing lease_id")?;
    let linkage = task
        .state
        .subagent_invocation()
        .cloned()
        .context("running subagent task missing durable invocation linkage")?;
    let parent_id = task
        .parent_id
        .as_ref()
        .context("subagent task missing parent_id")?;
    let parent = task_store
        .get(parent_id)
        .await
        .context("failed to read subagent parent task")?
        .with_context(|| format!("subagent parent task {parent_id} does not exist"))?;
    let spawn_index = task
        .spawn_index
        .context("running subagent task missing spawn_index")?;
    let pending_tool = pending_subagent_tool_call(&parent, spawn_index)?;
    ensure_confirm_tier_subagent_tool(&pending_tool)?;
    let subagent_name = canonical_subagent_name(&pending_tool.name).to_owned();

    Ok(SubagentTaskBootstrap {
        invocation_task: task.clone(),
        thread_id: task.thread_id.clone(),
        task_id: task.id.clone(),
        worker_id,
        lease_id,
        subagent_id: pending_tool.id,
        subagent_name,
        child_thread_id: linkage.child_thread_id,
        child_root_task_id: linkage.child_root_task_id,
        spec: linkage.spec,
    })
}

/// Materialize the final subagent result from terminal child-thread
/// state and complete the parent-visible invocation task.
///
/// # Errors
///
/// Returns an error if the child root task is missing or not
/// terminal, if the durable child-thread state cannot be loaded, if
/// the parent-facing result cannot be serialized, or if the
/// invocation task cannot be completed.
pub async fn execute_subagent_task(
    bootstrap: SubagentTaskBootstrap,
    deps: &SubagentResultDeps<'_>,
    now: OffsetDateTime,
) -> Result<SubagentTaskOutcome> {
    #[cfg(feature = "otel")]
    let started_at = std::time::Instant::now();

    #[cfg(feature = "otel")]
    crate::observability::ServerMetrics::global()
        .record_task_acquired(crate::observability::attrs::KIND_SUBAGENT);

    // Box-pin the inner future so the outer function's state
    // machine stays small enough that clippy::large_futures does
    // not fire on every test that drives subagent execution.
    let result = Box::pin(execute_subagent_task_inner(bootstrap, deps, now)).await;

    #[cfg(feature = "otel")]
    {
        let metrics = crate::observability::ServerMetrics::global();
        let outcome = match &result {
            Ok(_) => crate::observability::attrs::OUTCOME_DONE,
            Err(_) => crate::observability::attrs::OUTCOME_ERROR,
        };
        metrics.record_task_execution(
            crate::observability::attrs::KIND_SUBAGENT,
            outcome,
            started_at.elapsed().as_secs_f64(),
        );
    }

    result
}

/// Inner body of [`execute_subagent_task`]. Kept separate so the
/// outer function can wrap the entire call in a metric-recording
/// shim without threading a stopwatch through every early return.
async fn execute_subagent_task_inner(
    bootstrap: SubagentTaskBootstrap,
    deps: &SubagentResultDeps<'_>,
    now: OffsetDateTime,
) -> Result<SubagentTaskOutcome> {
    let subagent_result = materialize_terminal_subagent_result(&bootstrap, deps, now).await?;
    let tool_result = build_parent_tool_result(&subagent_result).context("build tool result")?;
    let result_payload =
        serde_json::to_value(&tool_result).context("serialize subagent tool result")?;
    let (invocation_task, parent_task) = deps
        .task_store
        .complete_task_with_result(
            &bootstrap.task_id,
            &bootstrap.worker_id,
            &bootstrap.lease_id,
            result_payload,
            now,
        )
        .await
        .context("complete subagent invocation task")?;
    let committed_events =
        commit_completed_subagent_progress(&bootstrap, &subagent_result.summary, deps, now).await;

    Ok(SubagentTaskOutcome {
        invocation_task,
        parent_task,
        subagent_result,
        tool_result,
        committed_events,
    })
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
/// `child_thread_id` must be pre-allocated by the caller and reused
/// across retries so [`ThreadStore::get_or_create`] is idempotent and
/// failed attempts do not orphan thread rows.
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
    mut spawn: SubagentInvocationSpawn,
    deps: &SubagentInvocationDeps<'_>,
    now: OffsetDateTime,
) -> Result<SpawnedSubagentInvocation> {
    let pending_tool_count = spawn.payload.continuation.payload.pending_tool_calls.len();
    let spawn_index_usize =
        usize::try_from(spawn.spawn_index).context("subagent spawn_index exceeds usize")?;
    ensure!(
        spawn_index_usize < pending_tool_count,
        "subagent spawn_index {spawn_index_usize} out of bounds for {pending_tool_count} pending tool calls",
    );
    let pending_tool =
        spawn.payload.continuation.payload.pending_tool_calls[spawn_index_usize].clone();
    ensure_confirm_tier_subagent_tool(&pending_tool)?;
    if spawn.child_root_input.is_empty() {
        spawn.child_root_input = build_child_root_input(&spawn.spec);
    }
    let child_thread = deps
        .thread_store
        .get_or_create(&spawn.child_thread_id, now)
        .await
        .context("materialize child thread projection")?;

    let (parent_task, invocation_task, child_root_task) = deps
        .task_store
        .spawn_subagent_invocation(parent_id, worker, lease, spawn, now)
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
    let subagent_name = canonical_subagent_name(&pending_tool.name);
    let started_event = build_parent_progress_event(&SubagentProgressSnapshot {
        subagent_id: &pending_tool.id,
        subagent_name,
        spec: &linkage.spec,
        child_thread_id: &child_thread.thread_id,
        child_root_task_id: &child_root_task.id,
        subagent_task_id: &invocation_task.id,
        completed: false,
        success: false,
        current_turn: 0,
        tool_count: 0,
        total_tokens: 0,
    });
    let committed_events = commit_parent_subagent_progress_if_possible(
        deps.event_repo,
        &parent_task.thread_id,
        started_event,
        now,
        "spawn",
        &pending_tool.id,
    )
    .await;

    Ok(SpawnedSubagentInvocation {
        parent_task,
        invocation_task,
        child_thread,
        child_root_task,
        committed_events,
    })
}

/// Records returned by [`spawn_subagent_batch_invocations`].
///
/// One [`SpawnedSubagentInvocation`] per entry in the input `spawns`,
/// in the same order. The first entry's `parent_task` is the
/// authoritative post-spawn parent row; subsequent entries carry a
/// clone for convenience so callers can iterate uniformly.
#[derive(Debug)]
pub struct SpawnedSubagentBatch {
    /// Parent task post-transition (`WaitingOnChildren`, with
    /// `pending_child_count == invocations.len()`).
    pub parent_task: AgentTask,
    /// One persisted invocation + child thread per entry, in input order.
    pub invocations: Vec<SpawnedSubagentInvocation>,
}

/// One entry in a fan-out subagent batch.
///
/// Unlike the store-level [`SubagentInvocationSpawn`], this carries no
/// `SuspensionPayload`: every entry in a batch shares the *same* parent
/// suspension, so it is passed exactly once to
/// [`spawn_subagent_batch_invocations`] — mirroring the store primitive
/// [`AgentTaskStore::spawn_subagent_batch`], which already takes the shared
/// payload explicitly. Removing the
/// per-entry payload makes the previously-implicit "every entry's
/// payload equals `spawns[0]`" assumption unrepresentable, so a
/// divergent payload from an external caller can no longer be silently
/// dropped.
#[derive(Clone, Debug)]
pub struct SubagentBatchEntry {
    /// Pre-allocated child-thread identifier (reused across retries).
    pub child_thread_id: ThreadId,
    /// Authoritative resolved spec for the child.
    pub spec: EffectiveSubagentSpec,
    /// Initial input for the child's first root turn. Empty means
    /// "derive it from `spec.prompt + spec.task`".
    pub child_root_input: Vec<SubmittedInputItem>,
    /// Index into the shared continuation's `pending_tool_calls`.
    pub spawn_index: u32,
    /// Host-supplied caller metadata to stamp on the child's durable
    /// root turn (`None` when the caller supplies none). Propagated
    /// verbatim into the child's `SubagentInvocationSpawn`.
    pub child_caller_metadata: Option<serde_json::Value>,
}

/// Persist N durable subagent invocations atomically under one parent
/// transition.
///
/// This is the fan-out flavour of [`spawn_subagent_invocation`]. The
/// validation, child-thread materialization, and parent-progress
/// emission semantics match the single-shot path, repeated per
/// entry. The single store-side CAS guarantees the parent moves to
/// `WaitingOnChildren { pending_child_count: N }` exactly once — no
/// partial fan-out is observable from concurrent readers.
///
/// `entries` must be non-empty; every entry's `spawn_index` must point
/// at the matching `pending_tool_calls` slot in the shared `payload`'s
/// continuation envelope; every `child_thread_id` must be unique across
/// the batch. The `payload` (the parent suspension) is passed once and
/// applies to all entries.
///
/// # Errors
///
/// Returns an error if the input is empty, if any entry references an
/// out-of-bounds `spawn_index`, if any of the targeted tool calls is
/// not Confirm-tier, if child-thread materialization fails for any
/// entry, if the store-side `spawn_subagent_batch` rejects the call
/// (CAS, leaf-parent, duplicate child thread, duplicate invocation
/// id), or if the durable linkage returned by the store is inconsistent.
/// Validate every entry in a fan-out batch against the shared
/// continuation envelope before materializing any child threads.
///
/// Returns the per-entry pending-tool-call info in input order so
/// downstream phases (event emission) can reuse it without re-indexing
/// into the continuation each time.
///
/// One bad `spawn_index` rejects the whole batch up front so we don't
/// orphan child thread rows after partial materialization.
fn validate_batch_spawns(
    payload: &SuspensionPayload,
    entries: &[SubagentBatchEntry],
) -> Result<Vec<agent_sdk_foundation::PendingToolCallInfo>> {
    let pending_tool_count = payload.continuation.payload.pending_tool_calls.len();
    let mut pending_tools_by_entry: Vec<agent_sdk_foundation::PendingToolCallInfo> =
        Vec::with_capacity(entries.len());
    for entry in entries {
        let spawn_index_usize =
            usize::try_from(entry.spawn_index).context("subagent spawn_index exceeds usize")?;
        ensure!(
            spawn_index_usize < pending_tool_count,
            "subagent spawn_index {spawn_index_usize} out of bounds for {pending_tool_count} pending tool calls",
        );
        let pending_tool =
            payload.continuation.payload.pending_tool_calls[spawn_index_usize].clone();
        ensure_confirm_tier_subagent_tool(&pending_tool)?;
        pending_tools_by_entry.push(pending_tool);
    }
    Ok(pending_tools_by_entry)
}

/// Materialize the per-entry child-thread row for every spawn in the
/// batch, defaulting empty `child_root_input` to the spec-derived
/// shape so the child's first root turn always has at least one
/// `SubmittedInputItem`.
async fn materialize_batch_child_threads(
    entries: &mut [SubagentBatchEntry],
    deps: &SubagentInvocationDeps<'_>,
    now: OffsetDateTime,
) -> Result<Vec<Thread>> {
    let mut child_threads = Vec::with_capacity(entries.len());
    for entry in entries.iter_mut() {
        if entry.child_root_input.is_empty() {
            entry.child_root_input = build_child_root_input(&entry.spec);
        }
        let child_thread = deps
            .thread_store
            .get_or_create(&entry.child_thread_id, now)
            .await
            .context("materialize child thread projection")?;
        child_threads.push(child_thread);
    }
    Ok(child_threads)
}

/// Per-entry inputs required to assemble one `SpawnedSubagentInvocation`
/// from a store batch return.
///
/// Bundles the four parallel collections (prepared invocation/child
/// pairs, materialized child threads, original pending-tool infos)
/// into one borrow so the per-entry assembly helper has a stable
/// signature.
struct BatchEntryAssembly<'a> {
    parent_task: &'a AgentTask,
    invocation_task: AgentTask,
    child_root_task: AgentTask,
    child_thread: Thread,
    pending_tool: agent_sdk_foundation::PendingToolCallInfo,
}

/// Verify the durable linkage on one batch entry and emit its
/// `SubagentProgress` start event.
///
/// Mirrors the per-entry section of [`spawn_subagent_invocation`] for
/// the fan-out case: confirms the store materialized the same child
/// thread / child root the worker pre-allocated, then commits the
/// `started` event so observers see a `SubagentProgress` per child.
async fn build_batch_invocation(
    entry: BatchEntryAssembly<'_>,
    deps: &SubagentInvocationDeps<'_>,
    now: OffsetDateTime,
) -> Result<SpawnedSubagentInvocation> {
    let BatchEntryAssembly {
        parent_task,
        invocation_task,
        child_root_task,
        child_thread,
        pending_tool,
    } = entry;

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

    let subagent_name = canonical_subagent_name(&pending_tool.name);
    let started_event = build_parent_progress_event(&SubagentProgressSnapshot {
        subagent_id: &pending_tool.id,
        subagent_name,
        spec: &linkage.spec,
        child_thread_id: &child_thread.thread_id,
        child_root_task_id: &child_root_task.id,
        subagent_task_id: &invocation_task.id,
        completed: false,
        success: false,
        current_turn: 0,
        tool_count: 0,
        total_tokens: 0,
    });
    let committed_events = commit_parent_subagent_progress_if_possible(
        deps.event_repo,
        &parent_task.thread_id,
        started_event,
        now,
        "batch_spawn",
        &pending_tool.id,
    )
    .await;

    Ok(SpawnedSubagentInvocation {
        parent_task: parent_task.clone(),
        invocation_task,
        child_thread,
        child_root_task,
        committed_events,
    })
}

/// Persist N durable subagent invocations under one parent transition.
///
/// Mirrors [`spawn_subagent_invocation`] for the fan-out case: validate
/// every entry up front, materialize per-entry child threads, fire a
/// single `spawn_subagent_batch` against the store, then walk the
/// returned invocation/child pairs to verify durable linkage and emit
/// per-entry `SubagentProgress` start events.
///
/// `child_thread_id` on every entry must be pre-allocated by the
/// caller and reused across retries (same idempotency contract as
/// `spawn_subagent_invocation`). `payload` is the shared parent
/// suspension and applies to every entry.
///
/// # Errors
///
/// Returns an error if the input batch is empty, any entry's
/// `spawn_index` is out of bounds for the shared continuation,
/// any entry references a non-Confirm-tier tool, the store rejects
/// the batch (CAS, leaf-parent, duplicate child thread, duplicate
/// invocation id), or if the durable linkage returned by the store
/// is inconsistent.
pub async fn spawn_subagent_batch_invocations(
    parent_id: &AgentTaskId,
    worker: &WorkerId,
    lease: &LeaseId,
    mut entries: Vec<SubagentBatchEntry>,
    payload: SuspensionPayload,
    deps: &SubagentInvocationDeps<'_>,
    now: OffsetDateTime,
) -> Result<SpawnedSubagentBatch> {
    ensure!(!entries.is_empty(), "subagent batch must be non-empty");

    // Validate every entry against the shared continuation envelope
    // before we start materializing child threads. All entries share
    // the one `payload`, so `payload.continuation.payload.pending_tool_calls`
    // is the same list for everyone — one bad spawn_index should reject
    // the whole batch up front, not after we've created N-1 child threads.
    let pending_tools_by_entry = validate_batch_spawns(&payload, &entries)?;

    // Per-entry: derive the default child input if blank, and
    // materialize the child thread row before the store sees it.
    let child_threads = materialize_batch_child_threads(&mut entries, deps, now).await?;

    // Assemble the store-level spawns. The store primitive takes the
    // shared `payload` explicitly and ignores the per-entry payload
    // field, so we fill that (frozen, non-optional) field from the
    // shared payload — there is no per-entry payload to diverge.
    let spawns: Vec<SubagentInvocationSpawn> = entries
        .into_iter()
        .map(|entry| SubagentInvocationSpawn {
            child_thread_id: entry.child_thread_id,
            spec: entry.spec,
            child_root_input: entry.child_root_input,
            spawn_index: entry.spawn_index,
            child_caller_metadata: entry.child_caller_metadata,
            payload: payload.clone(),
        })
        .collect();

    let (parent_task, prepared) = deps
        .task_store
        .spawn_subagent_batch(parent_id, worker, lease, spawns, payload, now)
        .await
        .context("persist subagent batch invocation tasks")?;

    ensure!(
        prepared.len() == child_threads.len(),
        "spawn_subagent_batch returned {} invocations for {} child threads",
        prepared.len(),
        child_threads.len(),
    );
    ensure!(
        prepared.len() == pending_tools_by_entry.len(),
        "spawn_subagent_batch returned {} invocations for {} pending tools",
        prepared.len(),
        pending_tools_by_entry.len(),
    );

    let mut invocations = Vec::with_capacity(prepared.len());
    for ((invocation_task, child_root_task), (child_thread, pending_tool)) in prepared
        .into_iter()
        .zip(child_threads.into_iter().zip(pending_tools_by_entry))
    {
        let assembled = build_batch_invocation(
            BatchEntryAssembly {
                parent_task: &parent_task,
                invocation_task,
                child_root_task,
                child_thread,
                pending_tool,
            },
            deps,
            now,
        )
        .await?;
        invocations.push(assembled);
    }

    Ok(SpawnedSubagentBatch {
        parent_task,
        invocations,
    })
}

fn build_child_root_input(spec: &EffectiveSubagentSpec) -> Vec<SubmittedInputItem> {
    let text = if spec.prompt.is_empty() {
        spec.task.clone()
    } else {
        format!("{}\n\n{}", spec.prompt, spec.task)
    };
    vec![SubmittedInputItem::Text { text }]
}

async fn load_child_final_response(
    message_store: &dyn MessageProjectionStore,
    thread_id: &ThreadId,
) -> Result<String> {
    let history = message_store
        .get_history(thread_id)
        .await
        .context("load child message history")?;
    Ok(history
        .iter()
        .rev()
        .find(|message| message.role == llm::Role::Assistant)
        .and_then(|message| message.content.first_text())
        .unwrap_or("")
        .to_owned())
}

fn child_root_error(child_root: &AgentTask) -> Option<String> {
    match child_root.status {
        TaskStatus::Failed => Some(
            child_root
                .last_error
                .clone()
                .unwrap_or_else(|| "subagent execution failed".to_owned()),
        ),
        TaskStatus::Cancelled => Some("subagent execution was cancelled".to_owned()),
        _ => None,
    }
}

fn build_parent_tool_result(result: &SubagentResult) -> Result<ToolResult> {
    let data = serde_json::to_value(result).context("serialize SubagentResult")?;
    let output = if result.summary.success {
        result.final_response.clone()
    } else {
        result
            .error_details
            .clone()
            .unwrap_or_else(|| "subagent execution failed".to_owned())
    };
    Ok(ToolResult {
        success: result.summary.success,
        output,
        data: Some(data),
        documents: Vec::new(),
        duration_ms: Some(result.summary.duration_ms),
    })
}

struct SubagentProgressSnapshot<'a> {
    subagent_id: &'a str,
    subagent_name: &'a str,
    spec: &'a EffectiveSubagentSpec,
    child_thread_id: &'a ThreadId,
    child_root_task_id: &'a AgentTaskId,
    subagent_task_id: &'a AgentTaskId,
    completed: bool,
    success: bool,
    current_turn: u32,
    tool_count: u32,
    total_tokens: u64,
}

fn build_parent_progress_event(snapshot: &SubagentProgressSnapshot<'_>) -> AgentEvent {
    AgentEvent::SubagentProgress {
        subagent_id: snapshot.subagent_id.to_owned(),
        subagent_name: snapshot.subagent_name.to_owned(),
        nickname: snapshot.spec.nickname.clone(),
        child_thread_id: Some(snapshot.child_thread_id.clone()),
        child_root_task_id: Some(snapshot.child_root_task_id.to_string()),
        subagent_task_id: Some(snapshot.subagent_task_id.to_string()),
        max_turns: Some(snapshot.spec.max_turns),
        current_turn: Some(snapshot.current_turn),
        model: Some(snapshot.spec.model.clone()),
        tool_name: snapshot.subagent_name.to_owned(),
        tool_context: snapshot.spec.task.clone(),
        completed: snapshot.completed,
        success: snapshot.success,
        tool_count: snapshot.tool_count,
        total_tokens: snapshot.total_tokens,
    }
}

async fn commit_parent_subagent_progress(
    event_repo: &dyn EventRepository,
    parent_thread_id: &ThreadId,
    event: AgentEvent,
    now: OffsetDateTime,
) -> Result<Vec<CommittedEvent>> {
    Ok(vec![
        event_repo
            .commit_event(parent_thread_id, event, now)
            .await
            .context("commit parent subagent progress event")?,
    ])
}

async fn commit_parent_subagent_progress_if_possible(
    event_repo: &dyn EventRepository,
    parent_thread_id: &ThreadId,
    event: AgentEvent,
    now: OffsetDateTime,
    phase: &str,
    subagent_id: &str,
) -> Vec<CommittedEvent> {
    match commit_parent_subagent_progress(event_repo, parent_thread_id, event, now).await {
        Ok(events) => events,
        Err(error) => {
            log::warn!(
                phase,
                parent_thread:? = parent_thread_id,
                subagent_id,
                error:? = error;
                "Failed to commit parent subagent progress event"
            );
            Vec::new()
        }
    }
}

fn pending_subagent_tool_call(
    parent: &AgentTask,
    spawn_index: u32,
) -> Result<agent_sdk_foundation::PendingToolCallInfo> {
    let spawn_index = usize::try_from(spawn_index).context("subagent spawn_index exceeds usize")?;
    let continuation = match &parent.state {
        crate::journal::task_state::TaskState::WaitingOnChildren { continuation, .. }
        | crate::journal::task_state::TaskState::ReadyToResume { continuation, .. } => {
            &continuation.payload
        }
        other => {
            anyhow::bail!(
                "subagent parent task {} missing continuation state, got {other:?}",
                parent.id,
            );
        }
    };
    ensure!(
        spawn_index < continuation.pending_tool_calls.len(),
        "subagent spawn_index {spawn_index} out of bounds for {} parent pending tool calls",
        continuation.pending_tool_calls.len(),
    );
    Ok(continuation.pending_tool_calls[spawn_index].clone())
}

fn canonical_subagent_name(tool_name: &str) -> &str {
    tool_name.strip_prefix("subagent_").unwrap_or(tool_name)
}

fn subagent_total_tokens(usage: &TokenUsage) -> u64 {
    u64::from(usage.input_tokens) + u64::from(usage.output_tokens)
}

fn elapsed_ms(start: OffsetDateTime, end: OffsetDateTime) -> u64 {
    let millis = (end - start).whole_milliseconds();
    u64::try_from(millis.max(0)).unwrap_or(u64::MAX)
}

async fn materialize_terminal_subagent_result(
    bootstrap: &SubagentTaskBootstrap,
    deps: &SubagentResultDeps<'_>,
    now: OffsetDateTime,
) -> Result<SubagentResult> {
    let child_root = deps
        .task_store
        .get(&bootstrap.child_root_task_id)
        .await
        .context("read child root task")?
        .with_context(|| {
            format!(
                "child root task {} does not exist",
                bootstrap.child_root_task_id
            )
        })?;
    ensure!(
        child_root.status.is_terminal(),
        "child root {} is not terminal (status {:?})",
        child_root.id,
        child_root.status,
    );
    ensure!(
        child_root.thread_id == bootstrap.child_thread_id,
        "child root {} belongs to thread {} but invocation points at {}",
        child_root.id,
        child_root.thread_id,
        bootstrap.child_thread_id,
    );

    let child_thread = deps
        .thread_store
        .get(&bootstrap.child_thread_id)
        .await
        .context("read child thread projection")?
        .with_context(|| format!("child thread {} does not exist", bootstrap.child_thread_id))?;
    let thread_tasks = deps
        .task_store
        .list_by_thread(&bootstrap.child_thread_id)
        .await
        .context("list child thread tasks")?;

    let success = child_root.status == TaskStatus::Completed;
    let final_response = if success {
        load_child_final_response(deps.message_store, &bootstrap.child_thread_id)
            .await
            .context("load child final response")?
    } else {
        String::new()
    };
    let tool_count = u32::try_from(
        thread_tasks
            .iter()
            .filter(|task| {
                task.kind == TaskKind::ToolRuntime
                    && matches!(task.status, TaskStatus::Completed | TaskStatus::Failed)
            })
            .count(),
    )
    .context("child tool count exceeds u32")?;

    Ok(SubagentResult {
        final_response,
        summary: SubagentSummary {
            success,
            total_turns: child_thread.committed_turns,
            tool_count,
            total_usage: child_thread.total_usage.clone(),
            duration_ms: elapsed_ms(
                child_root.created_at,
                child_root.completed_at.unwrap_or(now),
            ),
        },
        child_thread_id: bootstrap.child_thread_id.clone(),
        child_root_task_id: bootstrap.child_root_task_id.clone(),
        subagent_task_id: bootstrap.task_id.clone(),
        error_details: child_root_error(&child_root),
    })
}

async fn commit_completed_subagent_progress(
    bootstrap: &SubagentTaskBootstrap,
    summary: &SubagentSummary,
    deps: &SubagentResultDeps<'_>,
    now: OffsetDateTime,
) -> Vec<CommittedEvent> {
    let completed_event = build_parent_progress_event(&SubagentProgressSnapshot {
        subagent_id: &bootstrap.subagent_id,
        subagent_name: &bootstrap.subagent_name,
        spec: &bootstrap.spec,
        child_thread_id: &bootstrap.child_thread_id,
        child_root_task_id: &bootstrap.child_root_task_id,
        subagent_task_id: &bootstrap.task_id,
        completed: true,
        success: summary.success,
        current_turn: summary.total_turns,
        tool_count: summary.tool_count,
        total_tokens: subagent_total_tokens(&summary.total_usage),
    });

    commit_parent_subagent_progress_if_possible(
        deps.event_repo,
        &bootstrap.thread_id,
        completed_event,
        now,
        "completion",
        &bootstrap.subagent_id,
    )
    .await
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
    for capability in &request.capabilities.allowlist {
        ensure!(
            !capability.trim().is_empty(),
            "capability allowlist cannot contain blank identifiers",
        );
    }
    if let Some(mcp) = request.mcp.as_ref()
        && let Some(allowlist) = mcp.allowlist.as_ref()
    {
        for server in allowlist {
            ensure!(
                !server.trim().is_empty(),
                "MCP allowlist cannot contain blank identifiers",
            );
        }
    }
    Ok(())
}

fn normalize_optional_string(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

const fn legacy_effective_profile_name() -> &'static str {
    "__legacy_effective__"
}

const fn legacy_unavailable_capability() -> &'static str {
    "__legacy_unavailable__"
}

const fn default_subagent_depth() -> u32 {
    1
}

fn ensure_confirm_tier_subagent_tool(
    tool_call: &agent_sdk_foundation::PendingToolCallInfo,
) -> Result<()> {
    ensure!(
        tool_call.tier == ToolTier::Confirm,
        "subagent spawn `{}` must remain confirm-tier in the server model",
        tool_call.name,
    );
    Ok(())
}
