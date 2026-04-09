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
pub use agent_sdk_core::{ExternalToolResult, ToolRuntime, TurnOptions, TurnOutcome};
/// Durable reconstruction contract for worker-context recovery.
pub use agent_sdk_tools::{
    DefaultContextFactory, ExecutionContextFactory, HostDependencies, ToolContextSeed,
};
pub use agent_sdk_tools::{
    EventAuthority, EventStore, InMemoryEventStore, LocalEventAuthority, StoredTurnEvents,
};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use agent_sdk_core::{
        AgentConfig, AgentEvent, Message, Role, ThreadId, ToolRuntime, TurnOptions, TurnOutcome,
    };
    use agent_sdk_providers::LlmProvider;
    use agent_sdk_tools::{
        AgentHooks, EventAuthority, EventStore, HostDependencies, InMemoryEventStore,
        LocalEventAuthority, MessageStore, StateStore, StoredTurnEvents, ToolContext,
        ToolContextSeed,
    };
    use tokio_util::sync::CancellationToken;

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

    /// Prove the external tool handoff types are reachable from the server
    /// crate and round-trip through JSON for durable persistence.
    #[test]
    fn external_tool_handoff_types_are_reachable_and_serializable() -> anyhow::Result<()> {
        use agent_sdk_core::{
            AgentContinuation, AgentState, ExternalToolResult, PendingToolCallInfo, TokenUsage,
            ToolResult,
        };

        // Construct a PendingToolCalls continuation (the handoff payload)
        let thread = ThreadId::new();
        let continuation = AgentContinuation {
            thread_id: thread.clone(),
            turn: 3,
            total_usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
            turn_usage: TokenUsage {
                input_tokens: 30,
                output_tokens: 20,
            },
            pending_tool_calls: vec![
                PendingToolCallInfo {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    display_name: "Read File".into(),
                    input: serde_json::json!({"path": "/tmp/foo.txt"}),
                    listen_context: None,
                },
                PendingToolCallInfo {
                    id: "call_2".into(),
                    name: "write_file".into(),
                    display_name: "Write File".into(),
                    input: serde_json::json!({"path": "/tmp/bar.txt", "content": "hello"}),
                    listen_context: None,
                },
            ],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread.clone()),
        };

        // Round-trip the continuation through JSON (server persistence)
        let json = serde_json::to_string(&continuation)?;
        let recovered: AgentContinuation = serde_json::from_str(&json)?;
        assert_eq!(recovered.thread_id, thread);
        assert_eq!(recovered.turn, 3);
        assert_eq!(recovered.pending_tool_calls.len(), 2);

        // Construct the external results
        let results = vec![
            ExternalToolResult {
                tool_call_id: "call_1".into(),
                result: ToolResult::success("file contents here"),
            },
            ExternalToolResult {
                tool_call_id: "call_2".into(),
                result: ToolResult::success("written 5 bytes"),
            },
        ];

        // Round-trip results through JSON
        let results_json = serde_json::to_string(&results)?;
        let recovered_results: Vec<ExternalToolResult> = serde_json::from_str(&results_json)?;
        assert_eq!(recovered_results.len(), 2);
        assert_eq!(recovered_results[0].tool_call_id, "call_1");
        assert!(recovered_results[0].result.success);
        assert_eq!(recovered_results[1].tool_call_id, "call_2");

        Ok(())
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

    /// Prove that a server worker can reconstruct a `ToolContext` from a
    /// durable seed and fresh host dependencies, then emit events that
    /// land in the store with correct thread/turn/sequence binding.
    #[tokio::test]
    async fn tool_context_reconstructed_from_seed_emits_events() -> anyhow::Result<()> {
        let store = std::sync::Arc::new(InMemoryEventStore::new());
        let thread = ThreadId::from_string("t-seed");

        let seed = ToolContextSeed {
            thread_id: thread.clone(),
            turn: 3,
            sequence_offset: 10,
            metadata: {
                let mut m = std::collections::HashMap::new();
                m.insert("user_id".into(), serde_json::json!("u-1"));
                m
            },
        };

        let deps = HostDependencies {
            event_store: std::sync::Arc::clone(&store) as _,
            cancel_token: CancellationToken::new(),
            subagent_semaphore: None,
        };

        let ctx: ToolContext<()> = ToolContext::from_seed(&seed, (), deps);

        // Metadata forwarded
        assert_eq!(ctx.metadata.get("user_id"), Some(&serde_json::json!("u-1")));

        // Emit two events
        ctx.emit_event(AgentEvent::text("m1", "hello")).await?;
        ctx.emit_event(AgentEvent::text("m2", "world")).await?;

        // Events land in the right thread and turn
        let turn = store
            .get_turn(&thread, 3)
            .await?
            .ok_or_else(|| anyhow::anyhow!("turn 3 not found"))?;
        assert_eq!(turn.events.len(), 2);
        assert_eq!(turn.events[0].sequence, 10);
        assert_eq!(turn.events[1].sequence, 11);
        Ok(())
    }

    /// The seed must round-trip through JSON so the server can persist it
    /// alongside task state and recover it on worker restart.
    #[test]
    fn seed_serialization_round_trip() -> anyhow::Result<()> {
        let original = ToolContextSeed {
            thread_id: ThreadId::from_string("t-persist"),
            turn: 7,
            sequence_offset: 99,
            metadata: {
                let mut m = HashMap::new();
                m.insert("user_id".into(), serde_json::json!("u-42"));
                m.insert("tags".into(), serde_json::json!(["a", "b"]));
                m
            },
        };
        let json = serde_json::to_string(&original)?;
        let recovered: ToolContextSeed = serde_json::from_str(&json)?;
        assert_eq!(recovered, original);
        Ok(())
    }

    /// Prove that a second worker can pick up sequencing where the first
    /// left off using the seed's `sequence_offset`.
    #[tokio::test]
    async fn worker_restart_preserves_sequence_continuity() -> anyhow::Result<()> {
        let store = std::sync::Arc::new(InMemoryEventStore::new());
        let thread = ThreadId::from_string("t-restart");

        // ── Worker 1: turn 1 ─────────────────────────────────────────
        let seed_1 = ToolContextSeed::first_turn(thread.clone());
        let deps_1 = HostDependencies {
            event_store: std::sync::Arc::clone(&store) as _,
            cancel_token: CancellationToken::new(),
            subagent_semaphore: None,
        };
        let ctx_1: ToolContext<()> = ToolContext::from_seed(&seed_1, (), deps_1);

        ctx_1.emit_event(AgentEvent::text("m1", "first")).await?;
        ctx_1.emit_event(AgentEvent::text("m2", "second")).await?;
        store.finish_turn(&thread, 1).await?;

        // Determine offset for next worker
        let t1 = store
            .get_turn(&thread, 1)
            .await?
            .ok_or_else(|| anyhow::anyhow!("turn 1 not found"))?;
        let next_offset = t1
            .events
            .last()
            .ok_or_else(|| anyhow::anyhow!("no events in turn 1"))?
            .sequence
            + 1;

        // ── Worker 2: turn 2 (simulates restart) ─────────────────────
        let seed_2 = ToolContextSeed {
            thread_id: thread.clone(),
            turn: 2,
            sequence_offset: next_offset,
            metadata: HashMap::default(),
        };
        let deps_2 = HostDependencies {
            event_store: std::sync::Arc::clone(&store) as _,
            cancel_token: CancellationToken::new(),
            subagent_semaphore: None,
        };
        let ctx_2: ToolContext<()> = ToolContext::from_seed(&seed_2, (), deps_2);

        ctx_2.emit_event(AgentEvent::text("m3", "third")).await?;
        store.finish_turn(&thread, 2).await?;

        // ── Verify continuous ordering ───────────────────────────────
        let all = store.get_events(&thread).await?;
        let seqs: Vec<u64> = all.iter().map(|e| e.sequence).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
        Ok(())
    }
}
