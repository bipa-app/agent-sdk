//! Server-owned agent definition and runtime policy types.
//!
//! [`AgentDefinition`] is the server-side replacement for the SDK-local
//! [`AgentConfig`](agent_sdk_foundation::AgentConfig). Every field is resolved
//! from durable task identity by the registry — no SDK-local defaults
//! leak into this structure.
//!
//! [`RuntimePolicy`] captures the execution-level knobs the server
//! controls: tool execution mode, durability guarantees, retry budget,
//! and streaming preference.
//!
//! [`ThinkingPolicy`] is a serde-stable representation of the extended
//! thinking configuration. The SDK's [`ThinkingConfig`](agent_sdk_foundation::llm::ThinkingConfig)
//! does not implement `Serialize`/`Deserialize`, so the server defines
//! its own enum to ensure round-trip durability.

use std::sync::Arc;

use agent_sdk_foundation::ToolRuntime;
use agent_sdk_foundation::llm::{Effort, Tool};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Per-turn tool filtering
// ─────────────────────────────────────────────────────────────────────

/// Context passed to [`AgentDefinition::tools_fn`] at the start of each
/// turn.
///
/// Carries the caller metadata that was attached to the task at
/// submission time (see [`crate::journal::task::AgentTask::caller_metadata`]).
/// The SDK does not interpret this value — it's an opaque
/// `serde_json::Value` that the application-level filter
/// deserializes into its own domain type (role, user kind, entry
/// point, etc.) to decide which tools this turn sees.
#[derive(Clone, Debug)]
pub struct ToolFilterContext {
    /// Opaque per-turn caller metadata. Deserialize into your own
    /// domain type inside `tools_fn`.
    pub caller_metadata: serde_json::Value,
}

/// Dynamic-dispatch tool-filter closure type.
///
/// Stored on [`AgentDefinition::tools_fn`] when per-turn tool
/// filtering is desired. Invoked from the worker's turn-composition
/// path immediately before the LLM request is built.
pub type ToolsFn = Arc<dyn Fn(&ToolFilterContext) -> Vec<Tool> + Send + Sync>;

// ─────────────────────────────────────────────────────────────────────
// Thinking policy
// ─────────────────────────────────────────────────────────────────────

/// Server-owned thinking policy for an agent.
///
/// Mirrors the SDK's `ThinkingMode` / `ThinkingConfig` in a fully
/// serializable form so it can be persisted alongside the definition.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum ThinkingPolicy {
    /// No extended thinking.
    #[default]
    Disabled,
    /// Enabled with an explicit token budget.
    Enabled { budget_tokens: u32 },
    /// Adaptive — the model decides how much to think.
    Adaptive { effort: Option<Effort> },
}

// ─────────────────────────────────────────────────────────────────────
// Runtime policy
// ─────────────────────────────────────────────────────────────────────

/// Server-owned execution policy governing how a turn is run.
///
/// This is the server's authoritative source for execution-level knobs.
/// Later Phase 4 slices consume these fields to configure the worker's
/// turn execution loop.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePolicy {
    /// How tool calls are dispatched.
    pub tool_runtime: ToolRuntime,
    /// Checkpoint at every critical boundary for crash recovery.
    pub strict_durability: bool,
    /// Maximum number of execution attempts before the task is failed.
    pub max_attempts: u32,
    /// Enable streaming LLM responses.
    pub streaming: bool,
}

impl RuntimePolicy {
    /// Server-appropriate defaults: external tool dispatch, strict
    /// durability, three attempts, no streaming.
    #[must_use]
    pub const fn server_default() -> Self {
        Self {
            tool_runtime: ToolRuntime::External,
            strict_durability: true,
            max_attempts: 3,
            streaming: false,
        }
    }
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self::server_default()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Agent definition
// ─────────────────────────────────────────────────────────────────────

/// The fully-resolved, server-owned definition of an agent.
///
/// This replaces [`AgentConfig`](agent_sdk_foundation::AgentConfig) for the
/// server execution path. Every field is deterministically resolved from
/// durable task identity by the [`AgentDefinitionRegistry`](super::registry::AgentDefinitionRegistry) —
/// no SDK-local defaults participate in the resolution.
///
/// The struct is `Serialize + Deserialize` so it can be persisted as
/// part of audit rows or checkpoint metadata. The optional
/// [`tools_fn`](Self::tools_fn) is skipped by serde (closures are
/// not serializable) — deserialized definitions use the static
/// [`tools`](Self::tools) list.
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    // ── Provider / model ─────────────────────────────────────────
    /// LLM provider identifier (e.g. `"anthropic"`, `"openai"`).
    pub provider: String,
    /// Resolved model identifier within the provider.
    pub model: String,

    // ── Agent behaviour ──────────────────────────────────────────
    /// The agent's system prompt.
    pub system_prompt: String,
    /// Maximum tokens per LLM response.
    pub max_tokens: u32,
    /// Static tool definitions. Used when [`tools_fn`](Self::tools_fn)
    /// is `None`, and as the fallback for deserialized definitions
    /// (checkpoint replay / audit records) where the closure could
    /// not be persisted.
    pub tools: Vec<Tool>,
    /// Optional per-turn tool-filter closure.
    ///
    /// When `Some`, the worker invokes this function at turn start
    /// with a [`ToolFilterContext`] derived from the task's
    /// [`caller_metadata`](crate::journal::task::AgentTask::caller_metadata).
    /// The returned `Vec<Tool>` is what the LLM sees — tools not in
    /// the returned list are effectively invisible for that turn.
    ///
    /// When `None`, the static [`tools`](Self::tools) list is used.
    ///
    /// Serde-skipped because `Arc<dyn Fn>` cannot be serialized. On
    /// deserialization (audit replay, checkpoint restore) this is
    /// `None` and the static `tools` fallback applies — which is
    /// correct for those code paths, since they do not compose a
    /// fresh LLM request.
    #[serde(skip)]
    pub tools_fn: Option<ToolsFn>,
    /// Extended thinking configuration.
    pub thinking: ThinkingPolicy,

    // ── Execution policy ─────────────────────────────────────────
    /// Server-owned execution policy.
    pub policy: RuntimePolicy,
}

// ─────────────────────────────────────────────────────────────────────
// Manual trait impls
// ─────────────────────────────────────────────────────────────────────
//
// `tools_fn: Option<Arc<dyn Fn>>` rules out deriving `Debug`,
// `PartialEq`, and `Eq`. We implement them manually, treating
// `tools_fn` as presence-only (closure identity is not part of the
// definition's semantic identity — two definitions are equal iff
// their data fields match).

impl std::fmt::Debug for AgentDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentDefinition")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("system_prompt", &self.system_prompt)
            .field("max_tokens", &self.max_tokens)
            .field("tools", &self.tools)
            .field(
                "tools_fn",
                &self.tools_fn.as_ref().map_or("None", |_| "Some(<closure>)"),
            )
            .field("thinking", &self.thinking)
            .field("policy", &self.policy)
            .finish()
    }
}

impl PartialEq for AgentDefinition {
    fn eq(&self, other: &Self) -> bool {
        // `tools_fn` is intentionally excluded from equality — closure
        // identity is not part of the definition's semantic identity.
        self.provider == other.provider
            && self.model == other.model
            && self.system_prompt == other.system_prompt
            && self.max_tokens == other.max_tokens
            && self.tools == other.tools
            && self.thinking == other.thinking
            && self.policy == other.policy
    }
}

impl Eq for AgentDefinition {}

impl AgentDefinition {
    /// Resolve the effective tool list for a turn.
    ///
    /// If a [`tools_fn`](Self::tools_fn) is set AND the task has
    /// `caller_metadata`, invoke the filter. Otherwise fall back to
    /// the static [`tools`](Self::tools) vec — the filter has no
    /// caller context to act on, so it would be meaningless.
    #[must_use]
    pub fn resolve_tools(&self, caller_metadata: Option<&serde_json::Value>) -> Vec<Tool> {
        match (&self.tools_fn, caller_metadata) {
            (Some(f), Some(metadata)) => f(&ToolFilterContext {
                caller_metadata: metadata.clone(),
            }),
            _ => self.tools.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::ToolTier;

    fn tool(name: &str) -> Tool {
        Tool {
            name: name.into(),
            description: format!("{name} tool"),
            input_schema: serde_json::json!({ "type": "object" }),
            display_name: name.into(),
            tier: ToolTier::Observe,
        }
    }

    fn definition_with_static_tools(tools: Vec<Tool>) -> AgentDefinition {
        AgentDefinition {
            provider: "anthropic".into(),
            model: "test-model".into(),
            system_prompt: "test".into(),
            max_tokens: 1024,
            tools,
            tools_fn: None,
            thinking: ThinkingPolicy::default(),
            policy: RuntimePolicy::default(),
        }
    }

    #[test]
    fn resolve_tools_returns_static_when_no_tools_fn() {
        let def = definition_with_static_tools(vec![tool("ping"), tool("pong")]);
        let resolved = def.resolve_tools(None);
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].name, "ping");
        assert_eq!(resolved[1].name, "pong");
    }

    #[test]
    fn resolve_tools_falls_back_to_static_when_metadata_absent() {
        // tools_fn set but no caller_metadata → static tools win.
        let def = AgentDefinition {
            tools_fn: Some(Arc::new(|_| vec![tool("from_filter")])),
            ..definition_with_static_tools(vec![tool("static")])
        };
        let resolved = def.resolve_tools(None);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].name, "static");
    }

    #[test]
    fn resolve_tools_delegates_to_tools_fn_when_metadata_present() {
        // Filter: expose `admin_only` when caller_metadata.role == "admin";
        // otherwise expose `public_only`.
        let tools_fn: ToolsFn = Arc::new(|ctx| {
            let is_admin = ctx
                .caller_metadata
                .get("role")
                .and_then(serde_json::Value::as_str)
                == Some("admin");
            if is_admin {
                vec![tool("admin_only")]
            } else {
                vec![tool("public_only")]
            }
        });

        let def = AgentDefinition {
            tools_fn: Some(tools_fn),
            ..definition_with_static_tools(vec![tool("fallback")])
        };

        let admin_meta = serde_json::json!({ "role": "admin" });
        let admin_tools = def.resolve_tools(Some(&admin_meta));
        assert_eq!(admin_tools.len(), 1);
        assert_eq!(admin_tools[0].name, "admin_only");

        let guest_meta = serde_json::json!({ "role": "guest" });
        let guest_tools = def.resolve_tools(Some(&guest_meta));
        assert_eq!(guest_tools.len(), 1);
        assert_eq!(guest_tools[0].name, "public_only");
    }

    #[test]
    fn definition_serde_round_trips_without_tools_fn() {
        // tools_fn is #[serde(skip)] — ensure the rest of the struct
        // round-trips and the deserialized tools_fn is None.
        let def = AgentDefinition {
            tools_fn: Some(Arc::new(|_| vec![tool("ghost")])),
            ..definition_with_static_tools(vec![tool("persistent")])
        };
        let json = serde_json::to_string(&def).expect("serialize");
        let restored: AgentDefinition = serde_json::from_str(&json).expect("deserialize");
        assert!(restored.tools_fn.is_none());
        assert_eq!(restored.tools.len(), 1);
        assert_eq!(restored.tools[0].name, "persistent");
        // restored.resolve_tools falls back to the static list.
        let resolved = restored.resolve_tools(None);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].name, "persistent");
    }

    #[test]
    fn partial_eq_ignores_tools_fn() {
        let a = definition_with_static_tools(vec![tool("x")]);
        let b = AgentDefinition {
            tools_fn: Some(Arc::new(|_| Vec::new())),
            ..definition_with_static_tools(vec![tool("x")])
        };
        assert_eq!(a, b);
    }
}
