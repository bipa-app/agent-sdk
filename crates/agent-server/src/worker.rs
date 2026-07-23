//! Workers: agent-definition resolution, root-task bootstrapping, and
//! tool-runtime child-task execution.
//!
//! This module defines the server-owned [`AgentDefinition`] that replaces
//! the SDK-local [`AgentConfig`](agent_sdk_foundation::AgentConfig) for the
//! server execution path. The [`AgentDefinitionRegistry`] trait provides
//! the lookup surface for deterministic, durable-task-driven resolution.
//!
//! # Root turn entry point
//!
//! [`resolve_bootstrap_context`] is the single function a worker calls
//! after acquiring a root-turn task. It validates the task, resolves
//! its definition from the registry, and produces a
//! [`WorkerBootstrapContext`] that later worker stages consume.
//!
//! # Tool-runtime entry point
//!
//! [`resolve_tool_bootstrap`] is the corresponding function for
//! tool-runtime child tasks. It reads the parent task's durable state
//! to reconstruct exactly one [`PendingToolCallInfo`], then
//! [`execute_tool_task`] drives the child to completion or failure
//! through the journal.
//!
//! # Guarded execution
//!
//! [`guarded_tool_execution`] wraps [`execute_tool_task`] with the
//! durable execution-intent guard. Side-effecting and resumable tools
//! must persist an [`ExecutionIntent`] before the executor callback
//! runs. If persistence fails, execution is blocked (fail-closed).
//! The [`classify_tool_effect`] helper determines a tool's
//! [`ToolEffectClass`] from its [`PendingToolCallInfo`] metadata.
//!
//! # Confirmation pause/resume
//!
//! [`pause_tool_for_confirmation`] pauses a running tool-runtime
//! child task for user confirmation. [`apply_confirmation_decision`]
//! handles approval, rejection, or timeout. On approval,
//! [`resume_confirmed_tool`] re-checks authoritative policy via
//! [`ConfirmationPolicy`] before executing through the guarded path.
//!
//! # Parent resume from durable child outcomes
//!
//! [`aggregate_child_outcomes`] reads child task rows from the
//! journal and maps each terminal child to a deterministic
//! `(tool_call_id, ToolResult)` pair. [`resume_from_children`]
//! ties aggregation to [`resume_root_turn`], providing the single
//! entry point for journal-driven parent resume after all child
//! tool tasks reach terminal states.
//!
//! # Durable subagent spawn contract
//!
//! [`resolve_subagent_spec`] is the server-authoritative entry point
//! for durable subagent spawn resolution. It takes a typed
//! [`SubagentSpawnRequest`], narrows it through inherited parent
//! constraints plus a [`SubagentSpawnPolicy`], and returns exactly one
//! [`EffectiveSubagentSpec`] that later stages can use when creating
//! child threads and invocation tasks.
//!
//! [`spawn_subagent_invocation`] turns that authoritative spec into
//! durable task/thread records.
//!
//! [`PendingToolCallInfo`]: agent_sdk_foundation::PendingToolCallInfo
//!
//! ```ignore
//! use agent_server::worker::*;
//!
//! // After task acquisition:
//! let ctx = resolve_bootstrap_context(task, &registry).await?;
//!
//! // Later phases use ctx.definition, ctx.task, ctx.worker_id, etc.
//! ```
//!
//! # Design decisions
//!
//! - **Server-owned types**: [`AgentDefinition`], [`RuntimePolicy`], and
//!   [`ThinkingPolicy`] are fully `Serialize + Deserialize` so they can
//!   be persisted in audit rows and checkpoint metadata. They do not
//!   reuse the SDK's `AgentConfig` or `ThinkingConfig` which are
//!   designed for local-agent ergonomics, not durable server semantics.
//!
//! - **Registry is a trait**: different deployments (tests, database,
//!   config file) plug into the same bootstrap path.
//!
//! - **Preconditions on the task**: bootstrap only accepts `RootTurn` +
//!   `Running` tasks. This prevents later slices from accidentally
//!   bootstrapping a child task or a queued root.

pub mod activity;
pub mod bootstrap;
pub mod compaction;
pub mod confirmation;
pub mod connectivity;
pub mod definition;
pub mod registry;
pub mod root_turn;
pub mod subagent;
pub mod subagent_spawn_selector;
pub mod tool_task;
pub mod user_input;

#[cfg(test)]
mod bootstrap_test;
#[cfg(test)]
mod compaction_integration_test;
#[cfg(test)]
mod confirmation_test;
#[cfg(test)]
mod event_commit_test;
#[cfg(test)]
mod event_system_regression_test;
#[cfg(test)]
mod guarded_execution_test;
#[cfg(test)]
mod multimodal_input_test;
#[cfg(test)]
mod mutation_safety_test;
#[cfg(test)]
mod nested_restart_regression;
#[cfg(test)]
mod replay_coverage_test;
#[cfg(test)]
mod root_turn_test;
#[cfg(test)]
mod streaming_edge_cases_test;
#[cfg(test)]
mod subagent_execution_test;
#[cfg(test)]
mod subagent_test;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tool_task_test;

pub use activity::{ActivityBeacon, ActivityTrackingEventRepo};
pub use bootstrap::{WorkerBootstrapContext, resolve_bootstrap_context};
pub use connectivity::{ConnectivityWait, ConnectivityWaitGuard, ConnectivityWaitRegistry};
pub use definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
pub use registry::{AgentDefinitionRegistry, InMemoryAgentDefinitionRegistry};
pub use root_turn::{
    FailRootTurnParams, RootStreamFailure, RootTurnDeps, RootTurnOutcome, aggregate_child_outcomes,
    best_effort_close_open_attempts, cancel_root_turn, execute_root_turn, fail_root_turn,
    fail_root_turn_leaving_attempts_open_with_reason, fail_root_turn_with_reason,
    resume_for_steering, resume_from_children, resume_from_question, resume_root_turn,
    revert_steering_wake, terminal_reason_for_root_error,
};
pub use subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
    InheritedSubagentConstraints, InheritedSubagentPolicy, MixedChildrenRequest,
    ServerSubagentSpawnPolicy, SpawnedMixedBatch, SpawnedSubagentBatch, SpawnedSubagentInvocation,
    SubagentBatchEntry, SubagentCapabilityProfile, SubagentCapabilityRequest,
    SubagentInvocationDeps, SubagentMcpRequest, SubagentProgressSnapshot, SubagentResult,
    SubagentResultDeps, SubagentSandboxMode, SubagentSandboxPolicy, SubagentSpawnPolicy,
    SubagentSpawnRequest, SubagentSummary, SubagentTaskBootstrap, SubagentTaskOutcome,
    build_parent_progress_event, canonical_subagent_name, execute_subagent_task,
    resolve_subagent_bootstrap, resolve_subagent_spec, spawn_mixed_children_invocations,
    spawn_subagent_batch_invocations, spawn_subagent_invocation,
};
pub use subagent_spawn_selector::{
    BatchRouting, NoopSubagentSpawnSelector, SubagentSpawnDecision, SubagentSpawnSelector,
    classify_batch,
};
pub use tool_task::{
    ToolEventCollector, ToolTaskBootstrap, ToolTaskOutcome, execute_tool_task,
    resolve_tool_bootstrap,
};
pub use user_input::{UserInput, user_input_from_submitted};

// Re-export durable execution intent from journal.
pub use crate::journal::execution_intent::{
    ExecutionIntent, ExecutionIntentStore, InMemoryExecutionIntentStore, IntentStatus, OperationId,
    RetryDecision, ToolEffectClass, check_retry_safety, classify_tool_effect,
    guarded_tool_execution,
};

// Re-export confirmation types and functions.
pub use confirmation::{
    CONFIRMATION_POLICY_DENIED_PREFIX, CONFIRMATION_REJECTED_PREFIX, CONFIRMATION_TIMEOUT_PREFIX,
    ConfirmationDecision, ConfirmationDecisionOutcome, ConfirmationPolicy,
    ConfirmationResumeOutcome, PolicyVerdict, apply_confirmation_decision,
    pause_tool_for_confirmation, resume_confirmed_tool,
};
