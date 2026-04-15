//! Server-owned agent definition and runtime policy types.
//!
//! [`AgentDefinition`] is the server-side replacement for the SDK-local
//! [`AgentConfig`](agent_sdk_core::AgentConfig). Every field is resolved
//! from durable task identity by the registry — no SDK-local defaults
//! leak into this structure.
//!
//! [`RuntimePolicy`] captures the execution-level knobs the server
//! controls: tool execution mode, durability guarantees, retry budget,
//! and streaming preference.
//!
//! [`ThinkingPolicy`] is a serde-stable representation of the extended
//! thinking configuration. The SDK's [`ThinkingConfig`](agent_sdk_core::llm::ThinkingConfig)
//! does not implement `Serialize`/`Deserialize`, so the server defines
//! its own enum to ensure round-trip durability.

use agent_sdk_core::ToolRuntime;
use agent_sdk_core::llm::{Effort, Tool};
use serde::{Deserialize, Serialize};

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
/// This replaces [`AgentConfig`](agent_sdk_core::AgentConfig) for the
/// server execution path. Every field is deterministically resolved from
/// durable task identity by the [`AgentDefinitionRegistry`](super::registry::AgentDefinitionRegistry) —
/// no SDK-local defaults participate in the resolution.
///
/// The struct is `Serialize + Deserialize` so it can be persisted as
/// part of audit rows or checkpoint metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Tool definitions available to the agent.
    pub tools: Vec<Tool>,
    /// Extended thinking configuration.
    pub thinking: ThinkingPolicy,

    // ── Execution policy ─────────────────────────────────────────
    /// Server-owned execution policy.
    pub policy: RuntimePolicy,
}
