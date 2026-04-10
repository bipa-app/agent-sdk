//! Phase 4 worker: agent-definition resolution and root-task bootstrapping.
//!
//! This module defines the server-owned [`AgentDefinition`] that replaces
//! the SDK-local [`AgentConfig`](agent_sdk_core::AgentConfig) for the
//! server execution path. The [`AgentDefinitionRegistry`] trait provides
//! the lookup surface for deterministic, durable-task-driven resolution.
//!
//! # Entry point
//!
//! [`resolve_bootstrap_context`] is the single function a worker calls
//! after acquiring a root-turn task. It validates the task, resolves
//! its definition from the registry, and produces a
//! [`WorkerBootstrapContext`] that later Phase 4 slices consume:
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
pub mod definition;
pub mod registry;

#[cfg(test)]
mod bootstrap_test;

pub use bootstrap::{WorkerBootstrapContext, resolve_bootstrap_context};
pub use definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
pub use registry::{AgentDefinitionRegistry, InMemoryAgentDefinitionRegistry};
