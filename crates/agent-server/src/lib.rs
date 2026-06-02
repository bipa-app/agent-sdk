//! # Agent Server
//!
//! Server-side agent orchestration, built on the narrow SDK crates
//! ([`agent_sdk_foundation`], [`agent_sdk_tools`], [`agent_sdk_providers`])
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
//! use agent_sdk_foundation::{TurnOptions, ToolRuntime};
//!
//! let server_options = TurnOptions {
//!     tool_runtime: ToolRuntime::External,
//!     strict_durability: true,
//! };
//! ```
//!
//! ## Server-Facing Contract (Phase 1)
//!
//! After Phase 1 closes, every [`TurnOutcome`] the runtime hands back
//! to a server carries a structured [`TurnSummary`] alongside the
//! legacy per-variant fields. The summary is the **authoritative**
//! server contract; the legacy fields (`total_turns`, `input_tokens`,
//! `output_tokens`, `turn_usage`, …) remain only so existing local
//! callers do not break.
//!
//! New server code should read summaries instead of the legacy fields:
//!
//! ```ignore
//! use agent_server::{TurnOutcome, TurnSummary};
//!
//! match run_turn_outcome {
//!     TurnOutcome::PendingToolCalls {
//!         tool_calls,
//!         continuation,
//!         summary,
//!         ..
//!     } => {
//!         persist_turn_row(&summary);            // <- authoritative
//!         schedule_tool_tasks(tool_calls);        // <- caller's responsibility
//!         store_continuation(*continuation);      // <- opaque handoff payload
//!     }
//!     other => {
//!         if let Some(summary) = other.summary() {
//!             persist_turn_row(summary);
//!         }
//!     }
//! }
//! ```
//!
//! [`TurnSummary`] captures:
//!
//! | Field | Purpose |
//! |-------|---------|
//! | `thread_id` / `turn` / `total_turns` | Self-describing identity |
//! | `turn_usage` / `total_usage` | Token accounting for billing |
//! | `provenance` (provider, model) | Audit rows survive provider rotations |
//! | `response_id` | Join against raw provider response logs |
//! | `stop_reason` | Branch on `end_turn` / `tool_use` / `refusal` without reparsing history |
//! | `tool_call_count` | Tool-dispatch billing and runaway detection |
//! | `duration_ms` | SLO dashboards and retry budget tuning |
//! | `tool_runtime` / `strict_durability` | Execution profile recorded on the row |
//!
//! The summary format is serde-stable (`snake_case` discriminants,
//! explicit field keys) so it can be persisted directly in durable
//! turn rows.
//!
//! ## Authoritative vs Convenience Behaviour
//!
//! The split between **authoritative** server behaviour and local
//! **convenience** wrapper behaviour is a deliberate Phase 1 outcome:
//!
//! | Area | Authoritative (server) | Convenience (local) |
//! |------|------------------------|----------------------|
//! | Outcome metadata | [`TurnSummary`] | Legacy variant fields on [`TurnOutcome`] (`input_tokens`, etc.) |
//! | Event sink | Caller-supplied [`EventStore`] seeded by the server | In-memory store created by the builder |
//! | Event sequencing | [`LocalEventAuthority`] seeded with the last offset via [`LocalEventAuthority::with_offset`] | Fresh counter per `run_turn` |
//! | Tool execution | [`ToolRuntime::External`] — caller schedules tasks | [`ToolRuntime::Inline`] — SDK runs tools directly |
//! | Durability | `strict_durability: true`, checkpoint on every boundary | `strict_durability: false`, best-effort checkpoints |
//! | Continuation payload | Versioned [`agent_sdk_foundation::ContinuationEnvelope`] persisted by the server | Passed back in memory |
//! | Tool audit | [`agent_sdk_tools::ToolAuditSink`] producing [`ToolAuditRecord`](agent_sdk_foundation::ToolAuditRecord) | `NoopAuditSink` |
//!
//! Anything in the "Authoritative" column is part of the server-facing
//! Phase 1 contract and later phases (`journal`, `workers`, `transport`,
//! `storage`) will build on it.  Anything in the "Convenience" column
//! may change between SDK versions to preserve a good local-agent
//! experience and should not be relied on for durable server
//! semantics.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`journal`] | Durable `agent_tasks` schema, root submission queue, FIFO promotion, lease acquisition, heartbeats, expiry sweeps, Phase 2.4's typed pause-state with journal-guarded `pause_on_children` / `pause_on_confirmation` / `resume_from_confirmation`, Phase 2.5's retry budget / fail-closed recovery matrix shared across acquisition and expiry paths, Phase 2.6's tool-runtime child-task orchestration (`spawn_tool_children` / `complete_task` / `fail_task`) plus deterministic cancellation tree, and Phase 3.1's **threads projection** — durable thread-level aggregates (`committed_turns`, `total_usage`) owned exclusively by the completed-turn commit path |on cascade (`cancel_tree`) with journal-driven parent resume triggers |
//! | [`worker`] | Phase 4 worker bootstrapping: server-owned [`AgentDefinition`] resolution, [`AgentDefinitionRegistry`] lookup surface, and [`WorkerBootstrapContext`] construction for root-turn tasks. Phase 5.1 adds [`ToolTaskBootstrap`] and [`execute_tool_task`] for tool-runtime child-task execution. Phase 5.3 adds [`pause_tool_for_confirmation`], [`apply_confirmation_decision`], and [`resume_confirmed_tool`] for durable confirmation handling with authoritative policy rechecks. Phase 7.1 adds durable subagent spawn contracts via [`SubagentSpawnRequest`], [`InheritedSubagentConstraints`], and [`resolve_subagent_spec`]. Phase 7.2 adds durable invocation-task / child-thread allocation via [`spawn_subagent_invocation`]. |
//!
//! ## Planned modules (not yet implemented)
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `transport` | HTTP / WebSocket ingress |
//! | `storage` | Persistent message & state stores |

#![forbid(unsafe_code)]

#[macro_use]
pub mod failpoints;

/// Crate-root re-export of `fail-rs` so the exported [`fail_point!`] macro
/// resolves the registry through `$crate::__fail_reexport` rather than a
/// bare `::fail` path. This lets the macro fire from **other** workspace
/// crates (e.g. `agent-service-host`'s durable committers) that enable
/// `agent-server/failpoints` without depending on `fail-rs` directly.
/// Hidden from docs; not part of the public API.
#[cfg(feature = "failpoints")]
#[doc(hidden)]
pub use fail as __fail_reexport;

pub mod journal;
#[cfg(feature = "otel")]
pub mod observability;
pub mod worker;

pub use journal::{
    AgentTask, AgentTaskId, AgentTaskStore, ChildSpawnSpec, FailureReason, InMemoryAgentTaskStore,
    InMemoryMessageProjectionStore, InMemoryThreadStore, LeaseId, MessageProjection,
    MessageProjectionError, MessageProjectionStore, RecoveryAction, RecoveryContext,
    RecoveryRecord, RootWorkerInputs, StagedMessageStore, StagedStateStore, StagedStores,
    SubagentInvocationState, TaskKind, TaskSchemaError, TaskState, TaskStatus, Thread,
    ThreadSchemaError, ThreadStatus, ThreadStore, WorkerId, build_root_worker_inputs,
    classify_recovery,
};

// Phase 5.5: tool audit and redaction re-exports.
pub use journal::{
    InMemoryToolAuditEventStore, REDACTED_MARKER, RedactionLevel, RedactionPolicy, ToolAuditEvent,
    ToolAuditEventId, ToolAuditEventKind, ToolAuditEventParams, ToolAuditEventStore, redact_error,
    redact_string, redact_value,
};

// Phase 6.1: durable event committer and thread-scoped sequencing.
pub use journal::{CommittedEvent, EventRepository, InMemoryEventRepository};

// Phase 6.3: replay API and race-free replay-to-live handoff.
pub use journal::{EventNotifier, EventReceiver, EventStream, StreamEvent, stream_events};

// Phase 6.4: live tail hub with per-subscriber bounded buffers,
// lag detection, and replay-required disconnect.
pub use journal::{LiveTailConfig, LiveTailEvent, LiveTailHub, LiveTailReceiver, SubscriberId};

// ── Re-exports that validate the dependency edges ────────────────────
//
// Each `use` below proves that the corresponding narrow crate is
// reachable from `agent-server`. If any edge breaks, this crate
// fails to compile — exactly the early-warning signal we want.

/// Core contract types (IDs, events, LLM messages, turn outcomes).
pub use agent_sdk_foundation;

/// Tool traits, registry, hooks, and store contracts.
pub use agent_sdk_tools;

/// LLM provider trait and streaming primitives.
pub use agent_sdk_providers;

/// Convenience re-export of the server execution options and event-store types.
pub use agent_sdk_foundation::{
    AuditProvenance, ExternalToolResult, StopReason, ToolRuntime, TurnOptions, TurnOutcome,
    TurnSummary,
};
/// Durable reconstruction contract for worker-context recovery.
pub use agent_sdk_tools::{
    DefaultContextFactory, ExecutionContextFactory, HostDependencies, ToolContextSeed,
};
pub use agent_sdk_tools::{
    EventAuthority, EventStore, InMemoryEventStore, LocalEventAuthority, StoredTurnEvents,
};

/// Phase 4 worker types: definition, registry, bootstrap context,
/// and root turn execution.
///
/// Phase 5.1 adds tool-runtime worker types: tool-task bootstrap and
/// execution. Phase 5.3 adds confirmation pause/resume and policy
/// recheck.
pub use worker::{
    AgentDefinition, AgentDefinitionRegistry, BatchRouting, CONFIRMATION_POLICY_DENIED_PREFIX,
    CONFIRMATION_REJECTED_PREFIX, CONFIRMATION_TIMEOUT_PREFIX, ConfirmationDecision,
    ConfirmationDecisionOutcome, ConfirmationPolicy, ConfirmationResumeOutcome,
    EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
    InMemoryAgentDefinitionRegistry, InheritedSubagentConstraints, InheritedSubagentPolicy,
    NoopSubagentSpawnSelector, PolicyVerdict, RootTurnDeps, RootTurnOutcome, RuntimePolicy,
    ServerSubagentSpawnPolicy, SpawnedSubagentInvocation, SubagentCapabilityProfile,
    SubagentCapabilityRequest, SubagentInvocationDeps, SubagentMcpRequest, SubagentResult,
    SubagentResultDeps, SubagentSandboxMode, SubagentSandboxPolicy, SubagentSpawnDecision,
    SubagentSpawnPolicy, SubagentSpawnRequest, SubagentSpawnSelector, SubagentSummary,
    SubagentTaskBootstrap, SubagentTaskOutcome, ThinkingPolicy, ToolTaskBootstrap, ToolTaskOutcome,
    WorkerBootstrapContext, apply_confirmation_decision, classify_batch, execute_root_turn,
    execute_subagent_task, execute_tool_task, pause_tool_for_confirmation,
    resolve_bootstrap_context, resolve_subagent_bootstrap, resolve_subagent_spec,
    resolve_tool_bootstrap, resume_confirmed_tool, spawn_subagent_invocation,
};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use agent_sdk_foundation::{
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
        use agent_sdk_foundation::{
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
                ..Default::default()
            },
            turn_usage: TokenUsage {
                input_tokens: 30,
                output_tokens: 20,
                ..Default::default()
            },
            pending_tool_calls: vec![
                PendingToolCallInfo {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    display_name: "Read File".into(),
                    tier: agent_sdk_foundation::types::ToolTier::Observe,
                    input: serde_json::json!({"path": "/tmp/foo.txt"}),
                    effective_input: serde_json::json!({"path": "/tmp/foo.txt"}),
                    listen_context: None,
                },
                PendingToolCallInfo {
                    id: "call_2".into(),
                    name: "write_file".into(),
                    display_name: "Write File".into(),
                    tier: agent_sdk_foundation::types::ToolTier::Confirm,
                    input: serde_json::json!({"path": "/tmp/bar.txt", "content": "hello"}),
                    effective_input: serde_json::json!({"path": "/tmp/bar.txt", "content": "hello"}),
                    listen_context: None,
                },
            ],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread.clone()),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
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
    /// Verify the versioned continuation envelope round-trips through JSON
    /// and validates version at unwrap time.
    #[test]
    fn continuation_envelope_round_trip_and_version_check() -> anyhow::Result<()> {
        use agent_sdk_foundation::{
            AgentContinuation, AgentState, CONTINUATION_VERSION, ContinuationEnvelope,
            PendingToolCallInfo, TokenUsage,
        };

        let thread = ThreadId::new();
        let continuation = AgentContinuation {
            thread_id: thread.clone(),
            turn: 5,
            total_usage: TokenUsage {
                input_tokens: 200,
                output_tokens: 100,
                ..Default::default()
            },
            turn_usage: TokenUsage {
                input_tokens: 40,
                output_tokens: 30,
                ..Default::default()
            },
            pending_tool_calls: vec![PendingToolCallInfo {
                id: "call_1".into(),
                name: "read_file".into(),
                display_name: "Read File".into(),
                tier: agent_sdk_foundation::types::ToolTier::Observe,
                input: serde_json::json!({"path": "/tmp/foo.txt"}),
                effective_input: serde_json::json!({"path": "/tmp/foo.txt"}),
                listen_context: None,
            }],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread.clone()),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        };

        let envelope = ContinuationEnvelope::wrap(continuation);
        assert_eq!(envelope.version, CONTINUATION_VERSION);

        // Round-trip through JSON
        let json = serde_json::to_string(&envelope)?;
        let recovered: ContinuationEnvelope = serde_json::from_str(&json)?;
        assert_eq!(recovered.version, CONTINUATION_VERSION);
        assert_eq!(recovered.payload.thread_id, thread);
        assert_eq!(recovered.payload.turn, 5);

        // Unwrap validates version
        let inner = recovered.unwrap_validated();
        assert!(inner.is_ok());

        // Unknown version is rejected
        let bad = ContinuationEnvelope {
            version: 99,
            payload: inner.unwrap(),
        };
        let err = bad.unwrap_validated();
        assert!(err.is_err());
        assert!(
            err.unwrap_err()
                .contains("Unsupported continuation version 99")
        );

        Ok(())
    }

    /// Verify that `effective_input` is preserved through serialization.
    #[test]
    fn effective_input_survives_serialization() -> anyhow::Result<()> {
        use agent_sdk_foundation::PendingToolCallInfo;

        let info = PendingToolCallInfo {
            id: "call_1".into(),
            name: "tool".into(),
            display_name: "Tool".into(),
            tier: agent_sdk_foundation::types::ToolTier::Observe,
            input: serde_json::json!({"raw": true}),
            effective_input: serde_json::json!({"raw": true, "enriched": "yes"}),
            listen_context: None,
        };

        let json = serde_json::to_string(&info)?;
        let recovered: PendingToolCallInfo = serde_json::from_str(&json)?;
        assert_eq!(
            recovered.effective_input,
            serde_json::json!({"raw": true, "enriched": "yes"})
        );
        assert_ne!(recovered.input, recovered.effective_input);
        Ok(())
    }

    // ========================================================================
    // Phase 1.7 — TurnSummary contract regression suite
    // ========================================================================
    //
    // These tests validate the `TurnSummary` server-facing outcome
    // contract from the *server* side of the dependency graph.  They
    // check:
    //
    //   * reachability through the server crate's public re-exports,
    //   * durable JSON serialization shape,
    //   * version-stable snake_case discriminants,
    //   * the accessor on `TurnOutcome` that extracts the summary,
    //   * pattern matching on every variant without drift.
    //
    // They are intentionally data-only (no async runtime work, no
    // `run_turn` execution) because `agent-server` does not depend on
    // the `agent-sdk` runtime crate. End-to-end `TurnSummary` flow
    // through `run_turn` is covered by the regression suite in
    // `agent-sdk` (`test_turn_summary_*`).

    /// Compile-time proof that `TurnSummary` and its friends are
    /// reachable through `agent_server::*` without reaching into the
    /// narrow SDK crates.
    #[test]
    fn turn_summary_is_reachable_through_server_reexports() {
        use crate::{AuditProvenance, StopReason, ToolRuntime, TurnOptions, TurnSummary};

        fn _assert_types(
            _summary: TurnSummary,
            _provenance: AuditProvenance,
            _reason: StopReason,
            _runtime: ToolRuntime,
            _options: TurnOptions,
        ) {
        }
    }

    fn sample_turn_summary() -> agent_sdk_foundation::TurnSummary {
        use agent_sdk_foundation::{
            AuditProvenance, StopReason, ThreadId, TokenUsage, ToolRuntime, TurnSummary,
        };

        TurnSummary {
            thread_id: ThreadId::from_string("t-server-summary"),
            turn: 3,
            total_turns: 3,
            turn_usage: TokenUsage {
                input_tokens: 150,
                output_tokens: 75,
                ..Default::default()
            },
            total_usage: TokenUsage {
                input_tokens: 450,
                output_tokens: 200,
                ..Default::default()
            },
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
            response_id: Some("msg_abc123".into()),
            stop_reason: Some(StopReason::ToolUse),
            tool_call_count: 2,
            duration_ms: 1_842,
            tool_runtime: ToolRuntime::External,
            strict_durability: true,
        }
    }

    /// The `TurnSummary` wire format is the **durable server
    /// contract**.  Verify every field survives JSON round-trip so
    /// future server phases can persist summaries directly.
    #[test]
    fn turn_summary_round_trips_through_json_from_server_crate() -> anyhow::Result<()> {
        use agent_sdk_foundation::TurnSummary;

        let original = sample_turn_summary();
        let json = serde_json::to_string(&original)?;
        let recovered: TurnSummary = serde_json::from_str(&json)?;
        assert_eq!(recovered, original);
        Ok(())
    }

    /// The JSON shape has to be stable — rename a field and this test
    /// fails loudly instead of silently breaking a durable audit row.
    #[test]
    fn turn_summary_wire_format_is_stable() -> anyhow::Result<()> {
        let summary = sample_turn_summary();
        let value = serde_json::to_value(&summary)?;

        // Top-level keys the server contract depends on.
        for key in [
            "thread_id",
            "turn",
            "total_turns",
            "turn_usage",
            "total_usage",
            "provenance",
            "response_id",
            "stop_reason",
            "tool_call_count",
            "duration_ms",
            "tool_runtime",
            "strict_durability",
        ] {
            assert!(
                value.get(key).is_some(),
                "TurnSummary wire format lost key `{key}` — this breaks the server contract"
            );
        }

        // Nested provenance keys (identical shape to AuditProvenance).
        assert_eq!(
            value["provenance"]["provider"],
            serde_json::json!("anthropic")
        );
        assert_eq!(
            value["provenance"]["model"],
            serde_json::json!("claude-sonnet-4-5-20250929")
        );

        // Snake-case tool-runtime and stop-reason discriminants match
        // the provider wire formats so dashboards and joins line up.
        assert_eq!(value["tool_runtime"], serde_json::json!("external"));
        assert_eq!(value["stop_reason"], serde_json::json!("tool_use"));

        // Response ID survives as a plain string (not wrapped in extra
        // {"Some": ...} structure).
        assert_eq!(value["response_id"], serde_json::json!("msg_abc123"));

        Ok(())
    }

    /// A `TurnOutcome::Done` that a server would receive from
    /// `run_turn` must carry the summary in a way pattern-matching can
    /// extract.  This is a compile-time + runtime proof that the
    /// variant shape is stable.
    #[test]
    fn server_can_pattern_match_summary_off_done_outcome() {
        use agent_sdk_foundation::{TokenUsage, TurnOutcome};

        let summary = sample_turn_summary();
        let outcome = TurnOutcome::Done {
            total_turns: 3,
            total_usage: TokenUsage {
                input_tokens: 450,
                output_tokens: 200,
                ..Default::default()
            },
            summary: summary.clone(),
        };

        // Pattern match to pull the summary off without naming the
        // legacy fields — this is how server code will read outcomes.
        let TurnOutcome::Done {
            summary: matched, ..
        } = &outcome
        else {
            panic!("Expected Done");
        };
        assert_eq!(matched, &summary);

        // Accessor is also available and hides the variant name.
        assert_eq!(outcome.summary(), Some(&summary));
    }

    /// Server code should be able to build a new turn summary from
    /// scratch for tests / fixtures without reaching into private
    /// fields.
    #[test]
    fn server_can_construct_turn_summary_via_new() {
        use agent_sdk_foundation::{
            AuditProvenance, ThreadId, ToolRuntime, TurnOptions, TurnSummary,
        };

        let options = TurnOptions {
            tool_runtime: ToolRuntime::External,
            strict_durability: true,
        };
        let provenance = AuditProvenance::new("openai", "gpt-5");
        let summary = TurnSummary::new(ThreadId::from_string("t-new"), 4, provenance, &options);

        assert_eq!(summary.turn, 4);
        assert_eq!(summary.tool_runtime, ToolRuntime::External);
        assert!(summary.strict_durability);
    }

    /// Every variant except `Error` carries a summary. Guard against
    /// accidental variant additions by explicitly matching each one.
    #[test]
    fn every_terminal_variant_carries_a_summary() {
        use agent_sdk_foundation::{
            AgentContinuation, AgentError, AgentState, ContinuationEnvelope, ThreadId, TokenUsage,
            TurnOutcome,
        };

        let summary = sample_turn_summary();
        let thread = ThreadId::from_string("t-variants");
        let empty_continuation = Box::new(ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        }));

        let variants = vec![
            TurnOutcome::NeedsMoreTurns {
                turn: 1,
                turn_usage: TokenUsage::default(),
                total_usage: TokenUsage::default(),
                summary: summary.clone(),
            },
            TurnOutcome::Done {
                total_turns: 1,
                total_usage: TokenUsage::default(),
                summary: summary.clone(),
            },
            TurnOutcome::AwaitingConfirmation {
                tool_call_id: "c".into(),
                tool_name: "n".into(),
                display_name: "D".into(),
                input: serde_json::json!({}),
                description: "d".into(),
                continuation: empty_continuation.clone(),
                summary: summary.clone(),
            },
            TurnOutcome::Refusal {
                total_turns: 1,
                total_usage: TokenUsage::default(),
                summary: summary.clone(),
            },
            TurnOutcome::Cancelled {
                total_turns: 1,
                total_usage: TokenUsage::default(),
                summary: summary.clone(),
            },
            TurnOutcome::PendingToolCalls {
                turn: 1,
                turn_usage: TokenUsage::default(),
                total_usage: TokenUsage::default(),
                tool_calls: Vec::new(),
                continuation: empty_continuation,
                summary: summary.clone(),
            },
        ];

        for variant in &variants {
            assert_eq!(
                variant.summary(),
                Some(&summary),
                "variant {variant:?} dropped its summary",
            );
        }

        // Error is the only variant without a summary.
        let error_outcome = TurnOutcome::Error(AgentError::new("boom", false));
        assert!(error_outcome.summary().is_none());
    }
}
