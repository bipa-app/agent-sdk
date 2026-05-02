//! Per-tool-call routing decision for durable subagent spawn.
//!
//! Phase 7.2 introduced [`spawn_subagent_invocation`](super::subagent::spawn_subagent_invocation)
//! as the durable creation path for one parent-visible invocation
//! task plus one child thread. It is exposed as a public helper but
//! the worker's own root-turn loop never calls it — every tool call
//! in a turn flows through
//! [`AgentTaskStore::spawn_tool_children`](crate::journal::store::AgentTaskStore::spawn_tool_children),
//! which always produces [`TaskKind::ToolRuntime`](crate::journal::task::TaskKind::ToolRuntime)
//! children regardless of the tool name.
//!
//! That works for primitive tools (read, bash, …) but it leaves
//! "subagent" tools (`subagent_explore`, `subagent_plan`, the bip
//! `parallel_explore` fan-out, etc.) without a durable child-thread
//! path. Callers that want those tool calls to spawn real durable
//! subagent invocations had to drive `spawn_subagent_invocation`
//! themselves against a synthetic parent task — which duplicates the
//! root-turn contract and immediately breaks recovery / restart
//! guarantees.
//!
//! [`SubagentSpawnSelector`] closes that gap. The worker consults
//! the selector once per tool call inside
//! [`suspend_at_tool_boundary`](super::root_turn) (and the matching
//! resume sibling), and routes through `spawn_subagent_invocation`
//! when the selector returns
//! [`SubagentSpawnDecision::SpawnAsSubagent`].
//!
//! # Single-call vs fan-out (this PR)
//!
//! The first cut intentionally restricts subagent routing to **batches
//! that contain exactly one subagent tool call**. That covers the
//! single-shot subagent surface immediately and keeps the durable
//! contract tight: `spawn_subagent_invocation`'s CAS pauses the
//! parent on a `Running → WaitingOnChildren` transition, so calling
//! it more than once for the same parent in a single turn would
//! fail. The fan-out story (one turn → N subagent invocations) is
//! tracked as a separate slice; until it lands, mixed batches and
//! pure-fan-out batches keep flowing through `spawn_tool_children`
//! and the executor handles them as regular tool-runtime children.
//! Selector implementations should therefore inspect the **whole**
//! batch (`tool_calls`) before deciding any single call's fate, not
//! just the call under inspection in isolation.
//!
//! # Default
//!
//! [`NoopSubagentSpawnSelector`] keeps the legacy semantics — every
//! tool call routes through `spawn_tool_children`. Wiring sites that
//! don't opt in (every existing test, every legacy host) keep
//! today's exact behavior.

use agent_sdk_core::{PendingToolCallInfo, ThreadId};
use async_trait::async_trait;

use super::subagent::{EffectiveSubagentSpec, SubagentSpawnRequest};
use crate::journal::task::SubmittedInputItem;

/// Per-tool-call decision returned by [`SubagentSpawnSelector`].
#[derive(Clone, Debug)]
pub enum SubagentSpawnDecision {
    /// Treat the call as a regular tool-runtime child. The worker
    /// will route it through
    /// [`AgentTaskStore::spawn_tool_children`](crate::journal::store::AgentTaskStore::spawn_tool_children).
    /// This is the default for every primitive and policy tool.
    SpawnAsTool,

    /// Treat the call as a durable subagent invocation. The worker
    /// will route it through
    /// [`spawn_subagent_invocation`](super::subagent::spawn_subagent_invocation)
    /// using the carried [`SubagentSpawnPlan`].
    ///
    /// The plan is boxed because [`SubagentSpawnPlan`] embeds an
    /// [`EffectiveSubagentSpec`] (~700 bytes) and clippy's
    /// `large_enum_variant` lint flags an inline carrier as wasteful
    /// for the much-more-common [`SpawnAsTool`] path. Boxing keeps
    /// the enum cheap to pass around without changing semantics.
    ///
    /// Only legal when this is the **only** subagent decision in the
    /// batch — the worker rejects mixed batches today (see the
    /// module docs). Implementations that want to enable a call must
    /// first check that the batch contains exactly one such call.
    SpawnAsSubagent { plan: Box<SubagentSpawnPlan> },
}

/// Authoritative inputs the worker hands to
/// [`spawn_subagent_invocation`](super::subagent::spawn_subagent_invocation).
///
/// Selector implementations must pre-resolve the
/// [`EffectiveSubagentSpec`] and pre-allocate `child_thread_id`
/// before returning this plan: the worker has no way to look up
/// inherited constraints or the host's `SubagentSpawnPolicy`.
/// Callers typically reuse their existing
/// [`resolve_subagent_spec`](super::subagent::resolve_subagent_spec)
/// pipeline before constructing this struct.
#[derive(Clone, Debug)]
pub struct SubagentSpawnPlan {
    /// Caller intent that produced [`Self::spec`]. Persisted on
    /// the durable invocation linkage for audit and replay.
    pub request: SubagentSpawnRequest,
    /// Authoritative resolved spec. The worker forwards this to
    /// `spawn_subagent_invocation` unchanged.
    pub spec: EffectiveSubagentSpec,
    /// Pre-allocated child-thread identifier. The worker passes
    /// this through to `spawn_subagent_invocation` so a retry after
    /// lease expiry reuses the same thread row instead of orphaning
    /// it.
    pub child_thread_id: ThreadId,
    /// Initial input persisted on the child's first root-turn task.
    /// Empty means "let `spawn_subagent_invocation` derive it from
    /// `spec.prompt + spec.task`".
    pub child_root_input: Vec<SubmittedInputItem>,
}

/// Decides, per pending tool call, whether the worker should route
/// it through the durable subagent spawn path or the regular
/// tool-runtime spawn path.
///
/// The selector is consulted once per turn for every batch of pending
/// tool calls. Implementations receive the whole batch (so they can
/// reject mixed batches up front) plus the parent thread id (for
/// state lookups). The output is one decision per input slot, in
/// the same order as the input.
///
/// Implementations must be deterministic with respect to their
/// inputs: the worker may consult the selector on a fresh attempt
/// after a lease expiry, and divergent decisions across attempts
/// would corrupt the durable linkage between the parent and its
/// children.
#[async_trait]
pub trait SubagentSpawnSelector: Send + Sync {
    /// Compute one [`SubagentSpawnDecision`] per pending tool call.
    ///
    /// # Errors
    ///
    /// Implementations should return an error only for genuinely
    /// unrecoverable conditions (e.g. an internal store read
    /// failure). Routing-level failures — "this is not a subagent
    /// tool", "the role is unknown", "the request is invalid" —
    /// should resolve to [`SubagentSpawnDecision::SpawnAsTool`] so
    /// the call falls through to the regular executor and surfaces
    /// the error as a tool result the LLM can see.
    async fn decide(
        &self,
        parent_thread_id: &ThreadId,
        tool_calls: &[PendingToolCallInfo],
    ) -> anyhow::Result<Vec<SubagentSpawnDecision>>;
}

/// Default selector that routes every call through the regular
/// tool-runtime spawn path.
///
/// Use this when a host has no subagent surface, or when a host
/// wants to opt out of durable subagent routing entirely. Every
/// pre-existing call site of [`RootTurnDeps`](super::root_turn::RootTurnDeps)
/// behaves as if a [`NoopSubagentSpawnSelector`] were attached, so
/// switching to [`Some(&NoopSubagentSpawnSelector)`] is a true no-op.
#[derive(Default, Clone, Copy, Debug)]
pub struct NoopSubagentSpawnSelector;

#[async_trait]
impl SubagentSpawnSelector for NoopSubagentSpawnSelector {
    async fn decide(
        &self,
        _parent_thread_id: &ThreadId,
        tool_calls: &[PendingToolCallInfo],
    ) -> anyhow::Result<Vec<SubagentSpawnDecision>> {
        Ok(vec![SubagentSpawnDecision::SpawnAsTool; tool_calls.len()])
    }
}

/// Routing summary for a single batch of pending tool calls.
///
/// Built by [`classify_batch`] from a per-call decision vector. The
/// worker uses it to pick exactly one of three branches:
///
/// 1. [`BatchRouting::AllTools`] — drop straight into the legacy
///    `spawn_tool_children` path, identical to pre-PR behaviour.
/// 2. [`BatchRouting::SingleSubagent`] — route through
///    `spawn_subagent_invocation` for the carried index. The other
///    slots in the batch are guaranteed empty by the contract.
/// 3. [`BatchRouting::UnsupportedMixedBatch`] — the selector asked
///    for at least one subagent spawn but the batch isn't a pure
///    single-shot; today this falls back to `spawn_tool_children`
///    and the executor handles the subagent calls as regular tools.
///    Tracked for telemetry / fan-out follow-up.
#[derive(Debug)]
pub enum BatchRouting {
    AllTools,
    SingleSubagent {
        spawn_index: usize,
        /// Boxed for the same `large_enum_variant` reason as
        /// [`SubagentSpawnDecision::SpawnAsSubagent`] — keeps the
        /// `AllTools` and `UnsupportedMixedBatch` arms (the common
        /// case) cheap to pass around.
        plan: Box<SubagentSpawnPlan>,
    },
    UnsupportedMixedBatch,
}

/// Classify a per-call decision vector into a single
/// [`BatchRouting`] verdict.
///
/// Pulled out as a free function so the policy is unit-testable
/// without standing up the full root-turn machinery.
#[must_use]
pub fn classify_batch(decisions: Vec<SubagentSpawnDecision>) -> BatchRouting {
    let subagent_indices: Vec<usize> = decisions
        .iter()
        .enumerate()
        .filter_map(|(idx, decision)| match decision {
            SubagentSpawnDecision::SpawnAsSubagent { .. } => Some(idx),
            SubagentSpawnDecision::SpawnAsTool => None,
        })
        .collect();

    match subagent_indices.as_slice() {
        [] => BatchRouting::AllTools,
        [only] if decisions.len() == 1 => {
            // Pull the plan out of the single decision. The index
            // check above guarantees this is the
            // `SpawnAsSubagent` arm.
            let mut decisions = decisions;
            match decisions.swap_remove(*only) {
                SubagentSpawnDecision::SpawnAsSubagent { plan } => BatchRouting::SingleSubagent {
                    spawn_index: 0,
                    plan,
                },
                SubagentSpawnDecision::SpawnAsTool => {
                    // Unreachable per the filter above, but we
                    // surface it as the safe default rather than
                    // panicking.
                    BatchRouting::AllTools
                }
            }
        }
        _ => BatchRouting::UnsupportedMixedBatch,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use agent_sdk_core::ToolTier;
    use serde_json::json;

    use super::super::subagent::{
        EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, InheritedSubagentPolicy,
        SubagentCapabilityProfile, SubagentCapabilityRequest, SubagentSandboxPolicy,
    };

    fn dummy_tool_call(name: &str) -> PendingToolCallInfo {
        PendingToolCallInfo {
            id: format!("call_{name}"),
            name: name.to_owned(),
            display_name: name.to_owned(),
            tier: ToolTier::Confirm,
            input: json!({"task": "x"}),
            effective_input: json!({"task": "x"}),
            listen_context: None,
        }
    }

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|s| (*s).to_owned()).collect()
    }

    fn dummy_plan(task: &str) -> SubagentSpawnPlan {
        let request = SubagentSpawnRequest::new(task, SubagentCapabilityRequest::new("research"));
        let inherited_policy = InheritedSubagentPolicy {
            default_model: "claude-sonnet-4-5-20250929".to_owned(),
            allowed_models: set(&["claude-sonnet-4-5-20250929"]),
            default_max_turns: 5,
            max_turns: 5,
            default_timeout_ms: 30_000,
            max_timeout_ms: 30_000,
            capability_profiles: BTreeMap::from([(
                "research".to_owned(),
                SubagentCapabilityProfile {
                    capabilities: set(&["read_file"]),
                    sandbox: SubagentSandboxPolicy::read_only(),
                    allowed_mcp_servers: BTreeSet::new(),
                },
            )]),
            allowed_capabilities: set(&["read_file"]),
            max_depth: 3,
            max_parallel_subagents: 1,
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: BTreeSet::new(),
            audit_provider: "anthropic".to_owned(),
        };
        let spec = EffectiveSubagentSpec {
            task: task.to_owned(),
            prompt: String::new(),
            model: "claude-sonnet-4-5-20250929".to_owned(),
            max_turns: 5,
            timeout_ms: 30_000,
            depth: 1,
            max_parallel_subagents: 0,
            nickname: None,
            sandbox: SubagentSandboxPolicy::read_only(),
            mcp: EffectiveSubagentMcpPolicy::default(),
            audit_provenance: None,
            inherited_policy,
            capabilities: EffectiveSubagentCapabilities {
                profile: "research".to_owned(),
                allowed: set(&["read_file"]),
            },
        };
        SubagentSpawnPlan {
            request,
            spec,
            child_thread_id: ThreadId::new(),
            child_root_input: Vec::new(),
        }
    }

    #[tokio::test]
    async fn noop_selector_routes_every_call_as_tool() {
        let selector = NoopSubagentSpawnSelector;
        let thread_id = ThreadId::new();
        let calls = vec![
            dummy_tool_call("read"),
            dummy_tool_call("subagent_explore"),
            dummy_tool_call("bash"),
        ];
        let decisions = selector.decide(&thread_id, &calls).await.unwrap();
        assert_eq!(decisions.len(), 3);
        for decision in &decisions {
            assert!(matches!(decision, SubagentSpawnDecision::SpawnAsTool));
        }
    }

    #[test]
    fn classify_batch_empty_is_all_tools() {
        let routing = classify_batch(Vec::new());
        assert!(matches!(routing, BatchRouting::AllTools));
    }

    #[test]
    fn classify_batch_all_tools_is_all_tools() {
        let routing = classify_batch(vec![
            SubagentSpawnDecision::SpawnAsTool,
            SubagentSpawnDecision::SpawnAsTool,
        ]);
        assert!(matches!(routing, BatchRouting::AllTools));
    }

    #[test]
    fn classify_batch_single_subagent_unwraps_plan() {
        let routing = classify_batch(vec![SubagentSpawnDecision::SpawnAsSubagent {
            plan: Box::new(dummy_plan("explore")),
        }]);
        match routing {
            BatchRouting::SingleSubagent { spawn_index, plan } => {
                assert_eq!(spawn_index, 0);
                assert_eq!(plan.request.task, "explore");
                assert_eq!(plan.spec.task, "explore");
            }
            other => panic!("expected SingleSubagent, got {other:?}"),
        }
    }

    #[test]
    fn classify_batch_subagent_plus_tool_is_unsupported() {
        let routing = classify_batch(vec![
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore")),
            },
            SubagentSpawnDecision::SpawnAsTool,
        ]);
        assert!(matches!(routing, BatchRouting::UnsupportedMixedBatch));
    }

    #[test]
    fn classify_batch_two_subagents_is_unsupported() {
        let routing = classify_batch(vec![
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-a")),
            },
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-b")),
            },
        ]);
        assert!(matches!(routing, BatchRouting::UnsupportedMixedBatch));
    }
}
