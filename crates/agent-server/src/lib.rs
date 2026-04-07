//! # Agent Server
//!
//! Server-side agent orchestration, built on the narrow SDK crates
//! ([`agent_sdk_core`], [`agent_sdk_tools`], [`agent_sdk_providers`])
//! rather than the public-facing façade.
//!
//! This crate is the first real *consumer* of the split SDK boundary and
//! exists early in the workspace so that SDK boundary work can be validated
//! against a concrete dependency graph instead of a hypothetical one.
//!
//! ## Planned modules (not yet implemented)
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `journal` | Durable turn-level event log |
//! | `workers` | Background agent execution |
//! | `transport` | HTTP / WebSocket ingress |
//! | `storage` | Persistent message & state stores |

#![forbid(unsafe_code)]

// ── Re-exports that validate the dependency edges ────────────────────
//
// Each `use` below proves that the corresponding narrow crate is
// reachable from `agent-server`. If any edge breaks, this crate
// fails to compile — exactly the early-warning signal we want.

/// Core contract types (IDs, events, LLM messages, turn outcomes).
pub use agent_sdk_core;

/// Tool traits, registry, hooks, and store contracts.
pub use agent_sdk_tools;

/// LLM provider trait and streaming primitives.
pub use agent_sdk_providers;

#[cfg(test)]
mod tests {
    use agent_sdk_core::{AgentConfig, AgentEvent, Message, Role, ThreadId, TurnOutcome};
    use agent_sdk_providers::LlmProvider;
    use agent_sdk_tools::{AgentHooks, MessageStore, StateStore};

    /// Compile-time proof that the server crate can reach all three SDK
    /// sub-crates and name their key traits / types.
    #[test]
    fn dependency_edges_are_wired() {
        // Core types
        fn _assert_core_types(
            _id: ThreadId,
            _cfg: AgentConfig,
            _msg: Message,
            _role: Role,
        ) {
        }

        // Tool-layer traits (object-safe)
        fn _assert_tool_traits(
            _hooks: &dyn AgentHooks,
            _msgs: &dyn MessageStore,
            _state: &dyn StateStore,
        ) {
        }

        // Provider trait
        fn _assert_provider_trait(_p: &dyn LlmProvider) {}

        // Event & outcome types
        fn _assert_event_types(_e: AgentEvent, _o: TurnOutcome) {}
    }
}
