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
//! * **Mixed** — a batch carrying both `SpawnAsSubagent` and
//!   `SpawnAsTool` decisions (an LLM coordinator routinely emits N
//!   subagent calls plus a stray `todo_write` in one turn). Routes
//!   through
//!   [`spawn_mixed_children_invocations`](super::subagent::spawn_mixed_children_invocations),
//!   which CASes the parent once and persists the subagent slots
//!   **and** the tool-runtime children in a single store mutation. The
//!   parent transitions to `WaitingOnChildren { pending_child_count: N + M }`
//!   and resumes when the last child of either kind reports back.
//!
//! Selectors still receive the **whole** batch (`tool_calls`) so a
//! decision can depend on its siblings — e.g. a host that caps the
//! number of concurrent subagents needs to see them all at once.
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
    /// routes it through
    /// [`spawn_subagent_invocation`](super::subagent::spawn_subagent_invocation)
    /// (single-shot),
    /// [`spawn_subagent_batch_invocations`](super::subagent::spawn_subagent_batch_invocations)
    /// (pure fan-out), or
    /// [`spawn_mixed_children_invocations`](super::subagent::spawn_mixed_children_invocations)
    /// (alongside `SpawnAsTool` siblings) using the carried
    /// [`SubagentSpawnPlan`].
    ///
    /// The plan is boxed because [`SubagentSpawnPlan`] embeds an
    /// [`EffectiveSubagentSpec`] (~700 bytes) and clippy's
    /// `large_enum_variant` lint flags an inline carrier as wasteful
    /// for the much-more-common [`SubagentSpawnDecision::SpawnAsTool`]
    /// path. Boxing keeps the enum cheap to pass around without
    /// changing semantics.
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
    /// Opaque caller identity/spec projection for the child ROOT turn.
    /// A host `AgentDefinitionRegistry` reads this from the child root's
    /// `caller_metadata` to resolve a per-subagent model/toolset/prompt
    /// instead of falling back to the parent role. `None` reproduces the
    /// prior behavior (child root has no caller metadata). The SDK
    /// treats it as opaque JSON — it neither parses nor imposes a shape.
    pub child_caller_metadata: Option<serde_json::Value>,
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
///    `spawn_tool_children` path, identical to pre-selector behaviour.
/// 2. [`BatchRouting::SingleSubagent`] — route through
///    `spawn_subagent_invocation` for the carried index. The other
///    slots in the batch are guaranteed empty by the contract.
/// 3. [`BatchRouting::MultiSubagent`] — every decision in the batch
///    is `SpawnAsSubagent`. Routes through
///    `spawn_subagent_batch_invocations`, which CASes the parent once
///    and persists N invocation + child-thread pairs atomically. The
///    parent transitions to `WaitingOnChildren { pending_child_count: N }`
///    in the same store mutation.
/// 4. [`BatchRouting::Mixed`] — the batch carries both kinds. Routes
///    through `spawn_mixed_children_invocations`, which CASes the
///    parent once and persists the subagent slots **and** the
///    tool-runtime children in the same store mutation
///    (`pending_child_count = N + M`).
#[derive(Debug)]
pub enum BatchRouting {
    AllTools,
    SingleSubagent {
        spawn_index: usize,
        /// Boxed for the same `large_enum_variant` reason as
        /// [`SubagentSpawnDecision::SpawnAsSubagent`] — keeps the
        /// `AllTools` arm (the common case) cheap to pass around.
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
    /// Batch mixing at least one `SpawnAsSubagent` with at least one
    /// `SpawnAsTool`.
    ///
    /// `plans` and `tool_indices` partition the batch: together they
    /// name every slot of the turn's `pending_tool_calls` exactly once,
    /// which is what the resume-time positional fan-in
    /// ([`aggregate_child_outcomes`](super::root_turn::aggregate_child_outcomes))
    /// requires.
    Mixed {
        /// `(spawn_index, plan)` for every subagent slot, in input order.
        plans: Vec<(usize, Box<SubagentSpawnPlan>)>,
        /// Index of every slot that stays an ordinary tool child, in
        /// input order.
        tool_indices: Vec<usize>,
    },
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
    // Pure-subagent single-shot collapses to SingleSubagent so one
    // spawn doesn't pay the batch primitive's overhead.
    if subagent_count == total && total == 1 {
        let mut decisions = decisions;
        return match decisions.swap_remove(0) {
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
        };
    }

    let mut plans: Vec<(usize, Box<SubagentSpawnPlan>)> = Vec::with_capacity(subagent_count);
    let mut tool_indices: Vec<usize> = Vec::with_capacity(total - subagent_count);
    for (idx, decision) in decisions.into_iter().enumerate() {
        match decision {
            SubagentSpawnDecision::SpawnAsSubagent { plan } => plans.push((idx, plan)),
            SubagentSpawnDecision::SpawnAsTool => tool_indices.push(idx),
        }
    }

    if tool_indices.is_empty() {
        BatchRouting::MultiSubagent { plans }
    } else {
        BatchRouting::Mixed {
            plans,
            tool_indices,
        }
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
            child_caller_metadata: None,
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
    fn classify_batch_subagent_plus_tool_is_mixed() {
        let routing = classify_batch(vec![
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore")),
            },
            SubagentSpawnDecision::SpawnAsTool,
        ]);
        match routing {
            BatchRouting::Mixed {
                plans,
                tool_indices,
            } => {
                assert_eq!(plans.len(), 1);
                assert_eq!(plans[0].0, 0);
                assert_eq!(plans[0].1.request.task, "explore");
                assert_eq!(tool_indices, vec![1]);
            }
            other => panic!("expected Mixed, got {other:?}"),
        }
    }

    #[test]
    fn classify_batch_mixed_partitions_every_slot_exactly_once() {
        // The positional fan-in at resume time requires the union of
        // subagent slots and tool slots to cover pending_tool_calls
        // exactly once each.
        let routing = classify_batch(vec![
            SubagentSpawnDecision::SpawnAsTool,
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-a")),
            },
            SubagentSpawnDecision::SpawnAsTool,
            SubagentSpawnDecision::SpawnAsSubagent {
                plan: Box::new(dummy_plan("explore-b")),
            },
        ]);
        match routing {
            BatchRouting::Mixed {
                plans,
                tool_indices,
            } => {
                let subagent_slots: Vec<usize> = plans.iter().map(|(idx, _)| *idx).collect();
                assert_eq!(subagent_slots, vec![1, 3]);
                assert_eq!(tool_indices, vec![0, 2]);
                assert_eq!(plans[0].1.request.task, "explore-a");
                assert_eq!(plans[1].1.request.task, "explore-b");
            }
            other => panic!("expected Mixed, got {other:?}"),
        }
    }

    #[test]
    fn classify_batch_two_subagents_is_multi_subagent() {
        // Pure-subagent batch with N>=2 routes through the fan-out
        // path (MultiSubagent), not the single-shot one, and not the
        // mixed one — see classify_batch_subagent_plus_tool_is_mixed.
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
