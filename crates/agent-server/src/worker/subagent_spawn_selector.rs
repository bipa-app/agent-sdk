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
//! # Single-call vs fan-out
//!
//! The selector supports two flavours of subagent routing:
//!
//! * **Single-shot** — a batch containing exactly one
//!   `SpawnAsSubagent` decision (alone or alongside `SpawnAsTool`s
//!   ... see "mixed batches" below). Routes through
//!   [`spawn_subagent_invocation`](super::subagent::spawn_subagent_invocation).
//! * **Fan-out** — a batch where **every** decision is
//!   `SpawnAsSubagent` (i.e. the LLM emitted only subagent tool calls
//!   in this turn). Routes through
//!   [`spawn_subagent_batch_invocations`](super::subagent::spawn_subagent_batch_invocations),
//!   which performs one CAS on the parent and persists N
//!   invocation + child-thread pairs atomically. The parent transitions
//!   to `WaitingOnChildren { pending_child_count: N }` once and resumes
//!   when the last child reports back.
//!
//! **Mixed batches** — batches that contain both `SpawnAsTool` and
//! `SpawnAsSubagent` are routed as
//! [`BatchRouting::UnsupportedMixedBatch`] and fall back to
//! `spawn_tool_children`. The executor then handles the subagent
//! tool calls as regular tools so the LLM still sees a result rather
//! than a stuck turn. Splitting a mixed batch atomically (some children
//! `ToolRuntime`, some `Subagent`, all under one parent suspension) is
//! tracked as a future slice — it would need a new
//! `spawn_mixed_children` store primitive.
//!
//! Selector implementations therefore must inspect the **whole**
//! batch (`tool_calls`) before deciding any single call's fate, not
//! just the call under inspection in isolation.
//!
//! # Default
//!
//! [`NoopSubagentSpawnSelector`] keeps the legacy semantics — every
//! tool call routes through `spawn_tool_children`. Wiring sites that
//! don't opt in (every existing test, every legacy host) keep
//! today's exact behavior.

use agent_sdk_foundation::{PendingToolCallInfo, ThreadId};
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
    /// (single-shot) or
    /// [`spawn_subagent_batch_invocations`](super::subagent::spawn_subagent_batch_invocations)
    /// (fan-out — when every other decision in the batch is also
    /// `SpawnAsSubagent`) using the carried [`SubagentSpawnPlan`].
    ///
    /// The plan is boxed because [`SubagentSpawnPlan`] embeds an
    /// [`EffectiveSubagentSpec`] (~700 bytes) and clippy's
    /// `large_enum_variant` lint flags an inline carrier as wasteful
    /// for the much-more-common [`SubagentSpawnDecision::SpawnAsTool`]
    /// path. Boxing keeps the enum cheap to pass around without
    /// changing semantics.
    ///
    /// Mixed batches (`SpawnAsSubagent` alongside `SpawnAsTool`)
    /// degrade to [`BatchRouting::UnsupportedMixedBatch`] — the
    /// fan-out store primitive only takes pure subagent batches and
    /// the parent suspension envelope can't atomically split into
    /// `ToolRuntime` + `Subagent` children today. Selectors should
    /// inspect the whole batch before returning `SpawnAsSubagent`
    /// for any slot.
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
/// switching to `Some(&NoopSubagentSpawnSelector)` is a true no-op.
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
/// worker uses it to pick exactly one of four branches:
///
/// 1. [`BatchRouting::AllTools`] — drop straight into the legacy
///    `spawn_tool_children` path, identical to pre-PR behaviour.
/// 2. [`BatchRouting::SingleSubagent`] — route through
///    `spawn_subagent_invocation` for the carried index. The other
///    slots in the batch are guaranteed empty by the contract.
/// 3. [`BatchRouting::MultiSubagent`] — every decision in the batch
///    is `SpawnAsSubagent`. Routes through
///    `spawn_subagent_batch_invocations`, which CASes the parent once
///    and persists N invocation + child-thread pairs atomically. The
///    parent transitions to `WaitingOnChildren { pending_child_count: N }`
///    in the same store mutation.
/// 4. [`BatchRouting::UnsupportedMixedBatch`] — the selector asked
///    for at least one subagent spawn but the batch is genuinely
///    mixed (subagent calls alongside tool calls). Today this falls
///    back to `spawn_tool_children` and the executor handles the
///    subagent calls as regular tools so the LLM still sees a
///    result. Atomic mixed splits are tracked as a future slice.
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
    /// Pure-subagent batch with N >= 2 entries. Carries every
    /// `(spawn_index, plan)` pair in input order so the worker can
    /// build a `Vec<SubagentInvocationSpawn>` for
    /// [`spawn_subagent_batch_invocations`](super::subagent::spawn_subagent_batch_invocations).
    ///
    /// Single-entry pure-subagent batches collapse to
    /// [`BatchRouting::SingleSubagent`] (they don't need the
    /// fan-out store primitive's batch overhead) — `MultiSubagent`
    /// is reserved for genuine fan-out.
    MultiSubagent {
        plans: Vec<(usize, Box<SubagentSpawnPlan>)>,
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
    let total = decisions.len();
    let subagent_count = decisions
        .iter()
        .filter(|decision| matches!(decision, SubagentSpawnDecision::SpawnAsSubagent { .. }))
        .count();

    if subagent_count == 0 {
        return BatchRouting::AllTools;
    }
    if subagent_count != total {
        // Genuine mix: at least one tool call alongside at least one
        // subagent. The store primitive can't atomically split a
        // batch into ToolRuntime + Subagent children today, so we
        // degrade to AllTools at the call site (executor handles
        // the subagent tool calls as regular tools so the LLM still
        // sees a result).
        return BatchRouting::UnsupportedMixedBatch;
    }
    // Pure-subagent batch — single-shot (1 entry) collapses to
    // SingleSubagent so single spawns don't pay the batch primitive's
    // overhead, while N>=2 produces MultiSubagent for the fan-out
    // store path.
    if total == 1 {
        let mut decisions = decisions;
        match decisions.swap_remove(0) {
            SubagentSpawnDecision::SpawnAsSubagent { plan } => BatchRouting::SingleSubagent {
                spawn_index: 0,
                plan,
            },
            SubagentSpawnDecision::SpawnAsTool => {
                // Unreachable per the count check above, but we
                // surface it as the safe default rather than
                // panicking.
                BatchRouting::AllTools
            }
        }
    } else {
        let plans: Vec<(usize, Box<SubagentSpawnPlan>)> = decisions
            .into_iter()
            .enumerate()
            .filter_map(|(idx, decision)| match decision {
                SubagentSpawnDecision::SpawnAsSubagent { plan } => Some((idx, plan)),
                SubagentSpawnDecision::SpawnAsTool => None,
            })
            .collect();
        BatchRouting::MultiSubagent { plans }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use agent_sdk_foundation::ToolTier;
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
    fn classify_batch_two_subagents_is_multi_subagent() {
        // Pure-subagent batch with N>=2 routes through the fan-out
        // path (MultiSubagent), not the single-shot one. Tool calls
        // mixed with subagents still degrade to UnsupportedMixedBatch
        // — see classify_batch_subagent_plus_tool_is_unsupported.
        let routing = classify_batch(vec![
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-a")),
            },
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-b")),
            },
        ]);
        match routing {
            BatchRouting::MultiSubagent { plans } => {
                assert_eq!(plans.len(), 2);
                assert_eq!(plans[0].0, 0);
                assert_eq!(plans[0].1.request.task, "explore-a");
                assert_eq!(plans[1].0, 1);
                assert_eq!(plans[1].1.request.task, "explore-b");
            }
            other => panic!("expected MultiSubagent, got {other:?}"),
        }
    }

    #[test]
    fn classify_batch_three_subagents_is_multi_subagent_in_input_order() {
        // Indices must reflect the input position so the worker
        // can pair each plan with the matching pending_tool_call
        // slot in the parent's continuation envelope.
        let routing = classify_batch(vec![
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-a")),
            },
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-b")),
            },
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-c")),
            },
        ]);
        match routing {
            BatchRouting::MultiSubagent { plans } => {
                assert_eq!(plans.len(), 3);
                let tasks: Vec<_> = plans
                    .iter()
                    .map(|(idx, p)| (*idx, p.request.task.clone()))
                    .collect();
                assert_eq!(
                    tasks,
                    vec![
                        (0, "explore-a".to_string()),
                        (1, "explore-b".to_string()),
                        (2, "explore-c".to_string()),
                    ]
                );
            }
            other => panic!("expected MultiSubagent, got {other:?}"),
        }
    }
}
