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
//! ## Server Execution Model
//!
//! The server path uses [`TurnOptions`] to select server-appropriate
//! execution behaviour:
//!
//! - **[`ToolRuntime::External`]** — tool calls are returned to the caller
//!   for external execution instead of being run inline. This lets the
//!   server own tool-task dispatch and scheduling.
//! - **`strict_durability: true`** — state is checkpointed at every
//!   critical boundary (before/after LLM, before/after tools) so a
//!   crash at any point can be recovered.
//!
//! ```ignore
//! use agent_sdk_core::{TurnOptions, ToolRuntime};
//!
//! let server_options = TurnOptions {
//!     tool_runtime: ToolRuntime::External,
//!     strict_durability: true,
//! };
//! ```
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

/// Convenience re-export of the server execution options and event-store types.
pub use agent_sdk_core::{ToolRuntime, TurnOptions, TurnOutcome};
pub use agent_sdk_tools::{
    EventAuthority, EventStore, InMemoryEventStore, LocalEventAuthority, StoredTurnEvents,
};

#[cfg(test)]
mod tests {
    use agent_sdk_core::{
        AgentConfig, AgentEvent, Message, Role, ThreadId, ToolRuntime, TurnOptions, TurnOutcome,
    };
    use agent_sdk_providers::LlmProvider;
    use agent_sdk_tools::{
        AgentHooks, EventAuthority, EventStore, InMemoryEventStore, LocalEventAuthority,
        MessageStore, StateStore, StoredTurnEvents,
    };

    /// Compile-time proof that the server crate can reach all three SDK
    /// sub-crates and name their key traits / types.
    #[test]
    fn dependency_edges_are_wired() {
        // Core types
        fn _assert_core_types(_id: ThreadId, _cfg: AgentConfig, _msg: Message, _role: Role) {}

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

        // Event authority trait
        fn _assert_event_authority(_a: &dyn EventAuthority) {}
    }

    /// Prove the server crate can name the new execution-option types
    /// and construct server-appropriate defaults.
    #[test]
    fn server_execution_options_are_reachable() {
        let opts = TurnOptions {
            tool_runtime: ToolRuntime::External,
            strict_durability: true,
        };
        assert_eq!(opts.tool_runtime, ToolRuntime::External);
        assert!(opts.strict_durability);

        let store = InMemoryEventStore::new();
        let _: &dyn EventStore = &store;
        let _: StoredTurnEvents = StoredTurnEvents::default();
    }

    /// Demonstrate that a server can seed an authority with an offset and
    /// get continuous sequencing across multiple turns for the same thread.
    #[tokio::test]
    async fn seeded_authority_produces_continuous_sequences_across_turns() {
        let store = InMemoryEventStore::new();
        let thread = ThreadId::new();

        // ── Turn 1: authority starts at 0 ───────────────────────────
        let auth_t1 = LocalEventAuthority::new();
        store
            .append(&thread, 1, auth_t1.wrap(AgentEvent::text("m1", "hello")))
            .await
            .unwrap();
        store
            .append(&thread, 1, auth_t1.wrap(AgentEvent::text("m2", "world")))
            .await
            .unwrap();
        store.finish_turn(&thread, 1).await.unwrap();

        let turn_1 = store.get_turn(&thread, 1).await.unwrap().unwrap();
        assert_eq!(turn_1.events.len(), 2);
        assert_eq!(turn_1.events[0].sequence, 0);
        assert_eq!(turn_1.events[1].sequence, 1);

        // ── Turn 2: authority seeded at last_seq + 1 ────────────────
        let last_seq = turn_1.events.last().unwrap().sequence;
        let auth_t2 = LocalEventAuthority::with_offset(last_seq + 1);
        store
            .append(&thread, 2, auth_t2.wrap(AgentEvent::text("m3", "again")))
            .await
            .unwrap();
        store
            .append(&thread, 2, auth_t2.wrap(AgentEvent::text("m4", "!")))
            .await
            .unwrap();
        store.finish_turn(&thread, 2).await.unwrap();

        let turn_2 = store.get_turn(&thread, 2).await.unwrap().unwrap();
        assert_eq!(turn_2.events[0].sequence, 2);
        assert_eq!(turn_2.events[1].sequence, 3);

        // ── Verify global ordering ──────────────────────────────────
        let all = store.get_events(&thread).await.unwrap();
        let seqs: Vec<u64> = all.iter().map(|e| e.sequence).collect();
        assert_eq!(seqs, vec![0, 1, 2, 3]);
    }
}
