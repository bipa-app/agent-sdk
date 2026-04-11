//! Phase 4–5 workers: agent-definition resolution, root-task
//! bootstrapping, and tool-runtime child-task execution.
//!
//! This module defines the server-owned [`AgentDefinition`] that replaces
//! the SDK-local [`AgentConfig`](agent_sdk_core::AgentConfig) for the
//! server execution path. The [`AgentDefinitionRegistry`] trait provides
//! the lookup surface for deterministic, durable-task-driven resolution.
//!
//! # Root turn entry point
//!
//! [`resolve_bootstrap_context`] is the single function a worker calls
//! after acquiring a root-turn task. It validates the task, resolves
//! its definition from the registry, and produces a
//! [`WorkerBootstrapContext`] that later Phase 4 slices consume.
//!
//! # Tool-runtime entry point (Phase 5.1)
//!
//! [`resolve_tool_bootstrap`] is the corresponding function for
//! tool-runtime child tasks. It reads the parent task's durable state
//! to reconstruct exactly one [`PendingToolCallInfo`], then
//! [`execute_tool_task`] drives the child to completion or failure
//! through the journal.
//!
//! # Guarded execution (Phase 5.2)
//!
//! [`guarded_tool_execution`] wraps [`execute_tool_task`] with the
//! durable execution-intent guard. Side-effecting and resumable tools
//! must persist an [`ExecutionIntent`] before the executor callback
//! runs. If persistence fails, execution is blocked (fail-closed).
//! The [`classify_tool_effect`] helper determines a tool's
//! [`ToolEffectClass`] from its [`PendingToolCallInfo`] metadata.
//!
//! # Confirmation pause/resume (Phase 5.3)
//!
//! [`pause_tool_for_confirmation`] pauses a running tool-runtime
//! child task for user confirmation. [`apply_confirmation_decision`]
//! handles approval, rejection, or timeout. On approval,
//! [`resume_confirmed_tool`] re-checks authoritative policy via
//! [`ConfirmationPolicy`] before executing through the guarded path.
//!
//! # Parent resume from durable child outcomes (Phase 5.4)
//!
//! [`aggregate_child_outcomes`] reads child task rows from the
//! journal and maps each terminal child to a deterministic
//! `(tool_call_id, ToolResult)` pair. [`resume_from_children`]
//! ties aggregation to [`resume_root_turn`], providing the single
//! entry point for journal-driven parent resume after all child
//! tool tasks reach terminal states.
//!
//! [`PendingToolCallInfo`]: agent_sdk_core::PendingToolCallInfo
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

pub mod bootstrap;
pub mod confirmation;
pub mod definition;
pub mod registry;
pub mod root_turn;
pub mod tool_task;

#[cfg(test)]
mod bootstrap_test;
#[cfg(test)]
mod confirmation_test;
#[cfg(test)]
mod event_commit_test;
#[cfg(test)]
mod event_system_regression_test;
#[cfg(test)]
mod guarded_execution_test;
#[cfg(test)]
mod mutation_safety_test;
#[cfg(test)]
mod replay_coverage_test;
#[cfg(test)]
mod root_turn_test;
#[cfg(test)]
mod tool_task_test;

pub use bootstrap::{WorkerBootstrapContext, resolve_bootstrap_context};
pub use definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
pub use registry::{AgentDefinitionRegistry, InMemoryAgentDefinitionRegistry};
pub use root_turn::{
    RootTurnDeps, RootTurnOutcome, aggregate_child_outcomes, cancel_root_turn, execute_root_turn,
    fail_root_turn, resume_from_children, resume_root_turn,
};
pub use tool_task::{
    ToolEventCollector, ToolTaskBootstrap, ToolTaskOutcome, execute_tool_task,
    resolve_tool_bootstrap,
};

// Phase 5.2: re-export durable execution intent from journal.
pub use crate::journal::execution_intent::{
    ExecutionIntent, ExecutionIntentStore, InMemoryExecutionIntentStore, IntentStatus, OperationId,
    RetryDecision, ToolEffectClass, check_retry_safety, classify_tool_effect,
    guarded_tool_execution,
};

// Phase 5.3: re-export confirmation types and functions.
pub use confirmation::{
    CONFIRMATION_POLICY_DENIED_PREFIX, CONFIRMATION_REJECTED_PREFIX, CONFIRMATION_TIMEOUT_PREFIX,
    ConfirmationDecision, ConfirmationDecisionOutcome, ConfirmationPolicy,
    ConfirmationResumeOutcome, PolicyVerdict, apply_confirmation_decision,
    pause_tool_for_confirmation, resume_confirmed_tool,
};
