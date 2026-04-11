//! Store registry: owns and exposes the full set of durable store
//! trait-objects the journal and worker layers consume.
//!
//! [`StoreRegistry`] is the single struct that the service host hands to
//! workers, sweep tasks, and transport layers. It is constructed once
//! from [`StorageConfig`] and shared by reference for the lifetime of
//! the process.
//!
//! # Design rationale
//!
//! The journal and worker modules accept `&dyn Store` trait references
//! rather than concrete types. This registry centralises the concrete
//! instantiation so:
//!
//! 1. Callers never need to know which backend is active.
//! 2. Backend selection is a single match arm here.
//! 3. Mixed durability is explicit instead of hidden behind trait
//!    objects.
//!
//! [`StorageConfig`]: super::config::StorageConfig

use std::sync::Arc;

use anyhow::{Context, Result};

use agent_server::journal::checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
use agent_server::journal::event_notifier::EventNotifier;
use agent_server::journal::event_repository::{EventRepository, InMemoryEventRepository};
use agent_server::journal::execution_intent::{ExecutionIntentStore, InMemoryExecutionIntentStore};
use agent_server::journal::message_store::{
    InMemoryMessageProjectionStore, MessageProjectionStore,
};
use agent_server::journal::outbox::{InMemoryOutboxStore, OutboxStore};
use agent_server::journal::retention::{InMemoryRetentionStore, RetentionStore};
use agent_server::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
use agent_server::journal::thread_store::{InMemoryThreadStore, ThreadStore};
use agent_server::journal::tool_audit::{InMemoryToolAuditEventStore, ToolAuditEventStore};
use agent_server::journal::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
use agent_server::worker::registry::AgentDefinitionRegistry;

use super::config::{PostgresStorageConfig, StorageBackend, StorageConfig};
use super::postgres::store::PostgresDurableStore;

// ─────────────────────────────────────────────────────────────────────
// Durability surface report
// ─────────────────────────────────────────────────────────────────────

/// Per-surface durability summary for the active store registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageSurfaceStatus {
    /// Stable surface identifier used in logs, tests, and docs.
    pub surface: &'static str,
    /// Concrete backend supplying the surface.
    pub backend: &'static str,
    /// Whether the surface survives process restart.
    pub persists_restart: bool,
    /// Operator-facing note for hybrid/fallback surfaces.
    pub note: &'static str,
}

const IN_MEMORY_SURFACES: [StorageSurfaceStatus; 10] = [
    StorageSurfaceStatus {
        surface: "task_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "thread_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "message_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "attempt_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "checkpoint_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "event_repo",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "execution_intent_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "tool_audit_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "outbox_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
    StorageSurfaceStatus {
        surface: "retention_store",
        backend: "in_memory",
        persists_restart: false,
        note: "all state is process-local",
    },
];

const POSTGRES_SURFACES: [StorageSurfaceStatus; 10] = [
    StorageSurfaceStatus {
        surface: "task_store",
        backend: "postgres",
        persists_restart: true,
        note: "",
    },
    StorageSurfaceStatus {
        surface: "thread_store",
        backend: "postgres",
        persists_restart: true,
        note: "",
    },
    StorageSurfaceStatus {
        surface: "message_store",
        backend: "postgres",
        persists_restart: true,
        note: "",
    },
    StorageSurfaceStatus {
        surface: "attempt_store",
        backend: "postgres",
        persists_restart: true,
        note: "",
    },
    StorageSurfaceStatus {
        surface: "checkpoint_store",
        backend: "postgres",
        persists_restart: true,
        note: "",
    },
    StorageSurfaceStatus {
        surface: "event_repo",
        backend: "postgres",
        persists_restart: true,
        note: "",
    },
    StorageSurfaceStatus {
        surface: "execution_intent_store",
        backend: "in_memory",
        persists_restart: false,
        note: "execution intents remain process-local until a durable backend is implemented",
    },
    StorageSurfaceStatus {
        surface: "tool_audit_store",
        backend: "in_memory",
        persists_restart: false,
        note: "tool audit events remain process-local until a durable backend is implemented",
    },
    StorageSurfaceStatus {
        surface: "outbox_store",
        backend: "postgres",
        persists_restart: true,
        note: "",
    },
    StorageSurfaceStatus {
        surface: "retention_store",
        backend: "postgres",
        persists_restart: true,
        note: "",
    },
];

// ─────────────────────────────────────────────────────────────────────
// Backend state
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
enum RegistryBackend {
    InMemory,
    Postgres(PostgresBackend),
}

#[derive(Clone)]
struct PostgresBackend {
    store: Arc<PostgresDurableStore>,
    init_once: Arc<tokio::sync::OnceCell<()>>,
}

impl PostgresBackend {
    async fn initialize(&self) -> Result<()> {
        self.init_once
            .get_or_try_init(|| async {
                self.store
                    .migrate()
                    .await
                    .context("apply postgres durable-core migrations")
            })
            .await?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────

/// Owns every store the server needs, behind trait objects.
///
/// Constructed once at startup via [`StoreRegistry::from_config`] and
/// shared by reference throughout the process lifetime. Workers,
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
    /// Committed-event repository.
    pub event_repo: Arc<dyn EventRepository>,
    /// Execution-intent records for fail-closed guard.
    pub execution_intent_store: Arc<dyn ExecutionIntentStore>,
    /// Tool audit events.
    pub tool_audit_store: Arc<dyn ToolAuditEventStore>,
    /// Transactional outbox for durable event relay.
    pub outbox_store: Arc<dyn OutboxStore>,
    /// Retention-floor tracking for committed events.
    pub retention_store: Arc<dyn RetentionStore>,
    /// Agent definition lookup surface.
    pub definition_registry: Arc<dyn AgentDefinitionRegistry>,
    /// Same-process event notification hub for replay-to-live handoff.
    pub event_notifier: Arc<EventNotifier>,
    backend: RegistryBackend,
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
            StorageBackend::Postgres => {
                Self::postgres(config.postgres_settings()?, definition_registry)
            }
        }
    }

    /// Initialize backend-specific state such as migrations.
    ///
    /// This is idempotent and safe to call more than once.
    ///
    /// # Errors
    /// Returns an error if backend initialization fails.
    pub async fn initialize(&self) -> Result<()> {
        match &self.backend {
            RegistryBackend::InMemory => Ok(()),
            RegistryBackend::Postgres(backend) => backend.initialize().await,
        }
    }

    /// Stable backend label for logs and diagnostics.
    #[must_use]
    pub const fn backend_name(&self) -> &'static str {
        match self.backend {
            RegistryBackend::InMemory => "in_memory",
            RegistryBackend::Postgres(_) => "postgres",
        }
    }

    /// Per-surface durability report for the active backend.
    #[must_use]
    pub const fn durability_report(&self) -> &'static [StorageSurfaceStatus] {
        match self.backend {
            RegistryBackend::InMemory => &IN_MEMORY_SURFACES,
            RegistryBackend::Postgres(_) => &POSTGRES_SURFACES,
        }
    }

    /// Convenience constructor for the in-memory backend.
    ///
    /// Used by tests and local development. Every store is a fresh
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
            outbox_store: Arc::new(InMemoryOutboxStore::new()),
            retention_store: Arc::new(InMemoryRetentionStore::new()),
            definition_registry,
            event_notifier: Arc::new(EventNotifier::new()),
            backend: RegistryBackend::InMemory,
        }
    }

    fn postgres(
        config: &PostgresStorageConfig,
        definition_registry: Arc<dyn AgentDefinitionRegistry>,
    ) -> Result<Self> {
        let durable_store = build_postgres_store(config)?;

        Ok(Self {
            task_store: durable_store.clone(),
            thread_store: durable_store.clone(),
            message_store: durable_store.clone(),
            attempt_store: durable_store.clone(),
            checkpoint_store: durable_store.clone(),
            event_repo: durable_store.clone(),
            execution_intent_store: Arc::new(InMemoryExecutionIntentStore::new()),
            tool_audit_store: Arc::new(InMemoryToolAuditEventStore::new()),
            outbox_store: durable_store.clone(),
            retention_store: durable_store.clone(),
            definition_registry,
            event_notifier: Arc::new(EventNotifier::new()),
            backend: RegistryBackend::Postgres(PostgresBackend {
                store: durable_store,
                init_once: Arc::new(tokio::sync::OnceCell::new()),
            }),
        })
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

fn build_postgres_store(config: &PostgresStorageConfig) -> Result<Arc<PostgresDurableStore>> {
    let database_url = config.resolved_database_url()?;
    let max_connections = config.max_connections;
    let schema = config.schema.clone();
    let build = || {
        PostgresDurableStore::connect_lazy(&database_url, max_connections, schema.as_deref())
            .map(Arc::new)
    };

    if tokio::runtime::Handle::try_current().is_ok() {
        return build();
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build tokio runtime for postgres store bootstrap")?
        .block_on(async { build() })
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
        let _task = &stores.task_store;
        let _thread = &stores.thread_store;
        let _msg = &stores.message_store;
        let _attempt = &stores.attempt_store;
        let _ckpt = &stores.checkpoint_store;
        let _event = &stores.event_repo;
        let _intent = &stores.execution_intent_store;
        let _audit = &stores.tool_audit_store;
        let _outbox = &stores.outbox_store;
        let _retention = &stores.retention_store;
        let _def = &stores.definition_registry;
        let _notifier = &stores.event_notifier;
        assert_eq!(stores.backend_name(), "in_memory");
    }

    #[test]
    fn from_config_in_memory_matches_convenience() -> anyhow::Result<()> {
        let config = StorageConfig::default();
        let stores = StoreRegistry::from_config(&config, sample_registry())?;
        assert_eq!(stores.backend_name(), "in_memory");
        let _task = &stores.task_store;
        Ok(())
    }

    #[test]
    fn from_config_postgres_reports_remaining_non_durable_surfaces() -> anyhow::Result<()> {
        let config = StorageConfig {
            backend: StorageBackend::Postgres,
            postgres: PostgresStorageConfig {
                database_url: Some(
                    "postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk".into(),
                ),
                schema: Some("host_tests".into()),
                max_connections: 6,
            },
        };

        let stores = StoreRegistry::from_config(&config, sample_registry())?;
        assert_eq!(stores.backend_name(), "postgres");

        let task_surface = stores
            .durability_report()
            .iter()
            .find(|surface| surface.surface == "task_store")
            .context("missing task_store surface")?;
        assert!(task_surface.persists_restart);
        assert_eq!(task_surface.backend, "postgres");

        let event_surface = stores
            .durability_report()
            .iter()
            .find(|surface| surface.surface == "event_repo")
            .context("missing event_repo surface")?;
        assert!(event_surface.persists_restart);
        assert_eq!(event_surface.backend, "postgres");
        assert!(event_surface.note.is_empty());

        let outbox_surface = stores
            .durability_report()
            .iter()
            .find(|surface| surface.surface == "outbox_store")
            .context("missing outbox_store surface")?;
        assert!(outbox_surface.persists_restart);
        assert_eq!(outbox_surface.backend, "postgres");

        let nondurable_surfaces = stores
            .durability_report()
            .iter()
            .filter(|surface| !surface.persists_restart)
            .map(|surface| surface.surface)
            .collect::<Vec<_>>();
        assert_eq!(
            nondurable_surfaces,
            vec!["execution_intent_store", "tool_audit_store"]
        );
        Ok(())
    }

    #[test]
    fn root_turn_deps_borrows_compile() {
        let stores = StoreRegistry::in_memory(sample_registry());
        let deps = stores.root_turn_deps();
        let _ = deps.task_store;
        let _ = deps.thread_store;
    }

    #[test]
    fn registry_is_clone() {
        let stores = StoreRegistry::in_memory(sample_registry());
        let cloned = stores.clone();
        assert!(Arc::ptr_eq(&stores.task_store, &cloned.task_store));
    }
}
