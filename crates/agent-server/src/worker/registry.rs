//! Agent definition registry: the lookup surface that maps a runnable
//! task to its resolved [`AgentDefinition`].
//!
//! The registry is the single point of truth for runtime-policy
//! resolution. It is deliberately a trait so different deployment
//! topologies (in-memory tests, database-backed production) can plug
//! in without changing the worker bootstrap path.

use std::collections::HashMap;
use std::sync::RwLock;

use agent_sdk_foundation::ThreadId;
use anyhow::{Context, ensure};
use async_trait::async_trait;

use super::definition::AgentDefinition;
use crate::journal::task::{AgentTask, TaskKind};

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Resolves a runnable [`AgentTask`] to its [`AgentDefinition`].
///
/// Implementations must be deterministic: the same task identity must
/// always resolve to the same definition (barring explicit registry
/// mutations between calls). The resolution is durable-task-driven —
/// the task's own identity (thread, kind, root linkage) is the only
/// input the registry needs.
#[async_trait]
pub trait AgentDefinitionRegistry: Send + Sync {
    /// Resolve the definition for the given task.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The task is not a `RootTurn` (only root turns carry definitions).
    /// - No definition is registered for the task's identity.
    async fn resolve(&self, task: &AgentTask) -> anyhow::Result<AgentDefinition>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory implementation
// ─────────────────────────────────────────────────────────────────────

/// Test-friendly in-memory registry backed by a default definition and
/// optional per-thread overrides.
///
/// Resolution order:
/// 1. If a per-thread definition exists, return it.
/// 2. Otherwise, return the default definition.
pub struct InMemoryAgentDefinitionRegistry {
    default: AgentDefinition,
    by_thread: RwLock<HashMap<ThreadId, AgentDefinition>>,
}

impl InMemoryAgentDefinitionRegistry {
    #[must_use]
    pub fn new(default: AgentDefinition) -> Self {
        Self {
            default,
            by_thread: RwLock::new(HashMap::new()),
        }
    }

    /// Register a per-thread definition override.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock is poisoned.
    pub fn register_for_thread(
        &self,
        thread_id: ThreadId,
        definition: AgentDefinition,
    ) -> anyhow::Result<()> {
        self.by_thread
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(thread_id, definition);
        Ok(())
    }
}

#[async_trait]
impl AgentDefinitionRegistry for InMemoryAgentDefinitionRegistry {
    async fn resolve(&self, task: &AgentTask) -> anyhow::Result<AgentDefinition> {
        ensure!(
            task.kind == TaskKind::RootTurn,
            "only RootTurn tasks can be resolved to an AgentDefinition, got {:?}",
            task.kind,
        );

        let definition = self
            .by_thread
            .read()
            .ok()
            .context("lock poisoned")?
            .get(&task.thread_id)
            .unwrap_or(&self.default)
            .clone();
        Ok(definition)
    }
}
