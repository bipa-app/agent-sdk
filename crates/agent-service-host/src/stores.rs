//! Store registry: owns and exposes the full set of durable store
//! trait-objects the journal and worker layers consume.
//!
//! [`StoreRegistry`] is the single struct that the service host hands to
//! workers, sweep tasks, and (future) transport layers.  It is
//! constructed once from [`StorageConfig`] and shared by reference for
//! the lifetime of the process.
//!
//! # Design rationale
//!
//! The journal and worker modules accept `&dyn Store` trait references
//! rather than concrete types.  This registry centralises the concrete
//! instantiation so:
//!
//! 1. Callers never need to know which backend is active.
//! 2. Adding a new backend (e.g. Postgres) is a single match arm here.
//! 3. All stores share the same ownership model (`Arc`) so they can be
//!    cheaply cloned into background tasks.
//!
//! [`StorageConfig`]: super::config::StorageConfig

use std::sync::Arc;

use anyhow::Result;

use agent_server::journal::checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
use agent_server::journal::event_notifier::EventNotifier;
use agent_server::journal::event_repository::{EventRepository, InMemoryEventRepository};
use agent_server::journal::execution_intent::{ExecutionIntentStore, InMemoryExecutionIntentStore};
use agent_server::journal::message_store::{
    InMemoryMessageProjectionStore, MessageProjectionStore,
};
use agent_server::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use agent_server::journal::thread_store::{InMemoryThreadStore, ThreadStore};
use agent_server::journal::tool_audit::{InMemoryToolAuditEventStore, ToolAuditEventStore};
use agent_server::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
use agent_server::worker::registry::AgentDefinitionRegistry;

use super::config::{StorageBackend, StorageConfig};

// ─────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────

/// Owns every durable store the server needs, behind trait objects.
///
/// Constructed once at startup via [`StoreRegistry::from_config`] and
/// shared by reference throughout the process lifetime.  Workers,
/// sweep tasks, and transport layers access the stores they need
/// without knowing which concrete backend is active.
#[derive(Clone)]
pub struct StoreRegistry {
    /// Durable task journal.
    pub task_store: Arc<dyn AgentTaskStore>,
    /// Thread projection (committed turn aggregates).
    pub thread_store: Arc<dyn ThreadStore>,
    /// Message projection (ordered conversation messages).
    pub message_store: Arc<dyn MessageProjectionStore>,
    /// Turn-attempt audit log.
    pub attempt_store: Arc<dyn TurnAttemptStore>,
    /// Completed-turn checkpoints.
    pub checkpoint_store: Arc<dyn CheckpointStore>,
    /// Durable committed-event repository.
    pub event_repo: Arc<dyn EventRepository>,
    /// Execution-intent records for fail-closed guard.
    pub execution_intent_store: Arc<dyn ExecutionIntentStore>,
    /// Tool audit events.
    pub tool_audit_store: Arc<dyn ToolAuditEventStore>,
    /// Agent definition lookup surface.
    pub definition_registry: Arc<dyn AgentDefinitionRegistry>,
    /// Same-process event notification hub for replay-to-live handoff.
    pub event_notifier: Arc<EventNotifier>,
}

impl StoreRegistry {
    /// Construct the registry from a [`StorageConfig`], instantiating
    /// the correct backend for each store.
    ///
    /// The `definition_registry` is supplied separately because it is
    /// typically populated by the deployment layer (config file,
    /// database, API) rather than the storage backend.
    ///
    /// # Errors
    /// Returns an error if backend initialisation fails.
    pub fn from_config(
        config: &StorageConfig,
        definition_registry: Arc<dyn AgentDefinitionRegistry>,
    ) -> Result<Self> {
        match config.backend {
            StorageBackend::InMemory => Ok(Self::in_memory(definition_registry)),
        }
    }

    /// Convenience constructor for the in-memory backend.
    ///
    /// Used by tests and local development.  Every store is a fresh
    /// empty instance.
    #[must_use]
    pub fn in_memory(definition_registry: Arc<dyn AgentDefinitionRegistry>) -> Self {
        Self {
            task_store: Arc::new(InMemoryAgentTaskStore::new()),
            thread_store: Arc::new(InMemoryThreadStore::new()),
            message_store: Arc::new(InMemoryMessageProjectionStore::new()),
            attempt_store: Arc::new(InMemoryTurnAttemptStore::new()),
            checkpoint_store: Arc::new(InMemoryCheckpointStore::new()),
            event_repo: Arc::new(InMemoryEventRepository::new()),
            execution_intent_store: Arc::new(InMemoryExecutionIntentStore::new()),
            tool_audit_store: Arc::new(InMemoryToolAuditEventStore::new()),
            definition_registry,
            event_notifier: Arc::new(EventNotifier::new()),
        }
    }

    /// Build a [`agent_server::worker::root_turn::RootTurnDeps`] from
    /// the registry's stores.
    ///
    /// This is a convenience method that constructs the borrow-based
    /// deps struct the root-turn execution path expects.
    #[must_use]
    pub fn root_turn_deps(&self) -> agent_server::RootTurnDeps<'_> {
        agent_server::RootTurnDeps {
            task_store: self.task_store.as_ref(),
            thread_store: self.thread_store.as_ref(),
            message_store: self.message_store.as_ref(),
            attempt_store: self.attempt_store.as_ref(),
            checkpoint_store: self.checkpoint_store.as_ref(),
            event_repo: self.event_repo.as_ref(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
    use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;

    fn sample_definition() -> AgentDefinition {
        AgentDefinition {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5-20250929".into(),
            system_prompt: "test".into(),
            max_tokens: 4096,
            tools: Vec::new(),
            thinking: ThinkingPolicy::default(),
            policy: RuntimePolicy::server_default(),
        }
    }

    fn sample_registry() -> Arc<dyn AgentDefinitionRegistry> {
        Arc::new(InMemoryAgentDefinitionRegistry::new(sample_definition()))
    }

    #[test]
    fn in_memory_construction_succeeds() {
        let stores = StoreRegistry::in_memory(sample_registry());
        // Smoke test: all fields are populated (compilation is the
        // real proof; runtime just confirms no panics).
        let _task = &stores.task_store;
        let _thread = &stores.thread_store;
        let _msg = &stores.message_store;
        let _attempt = &stores.attempt_store;
        let _ckpt = &stores.checkpoint_store;
        let _event = &stores.event_repo;
        let _intent = &stores.execution_intent_store;
        let _audit = &stores.tool_audit_store;
        let _def = &stores.definition_registry;
        let _notifier = &stores.event_notifier;
    }

    #[test]
    fn from_config_in_memory_matches_convenience() {
        let config = StorageConfig::default();
        let stores = StoreRegistry::from_config(&config, sample_registry()).unwrap();
        let _task = &stores.task_store;
    }

    #[test]
    fn root_turn_deps_borrows_compile() {
        let stores = StoreRegistry::in_memory(sample_registry());
        let deps = stores.root_turn_deps();
        // Prove the deps struct fields are accessible.
        let _ = deps.task_store;
        let _ = deps.thread_store;
    }

    #[test]
    fn registry_is_clone() {
        let stores = StoreRegistry::in_memory(sample_registry());
        #[allow(clippy::redundant_clone)]
        let cloned = stores.clone();
        drop(cloned);
    }
}
