//! Typed configuration model for the service host.
//!
//! [`ServiceConfig`] is the single top-level struct loaded from YAML,
//! environment variables, or test fixtures.  Every nested section has
//! sensible defaults so a minimal config file works out of the box for
//! local development.
//!
//! # Defaults
//!
//! ```yaml
//! storage:
//!   backend: in_memory
//! worker:
//!   pool_size: 4
//!   lease_duration_secs: 30
//!   heartbeat_interval_secs: 10
//!   sweep_interval_secs: 5
//!   acquisition_interval_secs: 1
//! transport:
//!   grpc_enabled: false
//!   grpc_addr: "127.0.0.1:50051"
//!   http_enabled: false
//!   http_addr: "127.0.0.1:8080"
//! retention:
//!   event_ttl_secs: null
//!   checkpoint_max_per_thread: null
//! ```
//!
//! PostgreSQL-backed durable-core config:
//!
//! ```yaml
//! storage:
//!   backend: postgres
//!   postgres:
//!     database_url: postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk
//!     schema: agent_service_host
//!     max_connections: 8
//! ```

use std::net::SocketAddr;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Top-level config
// ─────────────────────────────────────��───────────────────────────────

/// Top-level service configuration.
///
/// Every section is optional at parse time and falls back to
/// [`Default`] values.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ServiceConfig {
    /// Storage backend selection.
    pub storage: StorageConfig,
    /// Worker pool sizing and lease management.
    pub worker: WorkerConfig,
    /// Transport enablement (gRPC, HTTP).
    pub transport: TransportConfig,
    /// Data retention policies.
    pub retention: RetentionConfig,
}

impl ServiceConfig {
    /// Load configuration from a YAML file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_yaml_file(path: &Path) -> Result<Self> {
        let contents =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_yaml_str(&contents)
    }

    /// Parse configuration from a YAML string.
    ///
    /// # Errors
    /// Returns an error if the YAML is malformed.
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).context("parsing service config YAML")
    }
}

// ─────────────────────────────────────────────────────────────────────
// Storage
// ─────────────────────────────────────────────────────────────────────

/// Which durable storage backend to use.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageBackend {
    /// All state lives in process memory.  Useful for local development
    /// and integration tests.  State is lost on restart.
    #[default]
    InMemory,
    /// Durable-core task, thread, message, attempt, and checkpoint
    /// state lives in `PostgreSQL`.
    Postgres,
    /// Embedded SQLite database in WAL mode.  Designed for desktop and
    /// CLI processes that own a single data directory.  State survives
    /// process restarts.
    ///
    /// Gated behind the `sqlite` cargo feature.
    Sqlite {
        /// Path to the SQLite database file.
        ///
        /// When `None`, the store uses a platform-default data directory:
        /// - Linux: `$XDG_DATA_HOME/agent-sdk/agent-sdk.db`
        /// - macOS: `~/Library/Application Support/agent-sdk/agent-sdk.db`
        /// - Windows: `%LOCALAPPDATA%\agent-sdk\agent-sdk.db`
        path: Option<String>,
    },
}

/// `PostgreSQL` backend settings.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PostgresStorageConfig {
    /// Connection string used for the durable-core tables.
    ///
    /// If omitted, the host falls back to the `DATABASE_URL`
    /// environment variable.
    pub database_url: Option<String>,
    /// Optional schema/search-path to use for the durable-core tables.
    pub schema: Option<String>,
    /// Maximum pooled Postgres connections for the durable-core store.
    pub max_connections: u32,
}

impl PostgresStorageConfig {
    pub(crate) const fn is_default(&self) -> bool {
        self.database_url.is_none() && self.schema.is_none() && self.max_connections == 8
    }

    /// Resolve the database URL for runtime use.
    ///
    /// # Errors
    /// Returns an error if neither `database_url` nor `DATABASE_URL`
    /// is available.
    pub fn resolved_database_url(&self) -> Result<String> {
        if let Some(url) = &self.database_url {
            return Ok(url.clone());
        }

        std::env::var("DATABASE_URL")
            .context("storage.postgres.database_url is unset and DATABASE_URL is not available")
    }
}

impl Default for PostgresStorageConfig {
    fn default() -> Self {
        Self {
            database_url: None,
            schema: None,
            max_connections: 8,
        }
    }
}

impl std::fmt::Debug for PostgresStorageConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresStorageConfig")
            .field(
                "database_url",
                &self.database_url.as_ref().map(|_| "<redacted>"),
            )
            .field("schema", &self.schema)
            .field("max_connections", &self.max_connections)
            .finish()
    }
}

/// Storage configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// The backend to instantiate.
    pub backend: StorageBackend,
    /// PostgreSQL-specific settings.
    #[serde(skip_serializing_if = "PostgresStorageConfig::is_default")]
    pub postgres: PostgresStorageConfig,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::InMemory,
            postgres: PostgresStorageConfig::default(),
        }
    }
}

impl StorageConfig {
    /// Return the Postgres settings when the backend is selected.
    ///
    /// # Errors
    /// Returns an error if the backend is not `postgres`.
    pub fn postgres_settings(&self) -> Result<&PostgresStorageConfig> {
        match self.backend {
            StorageBackend::Postgres => Ok(&self.postgres),
            StorageBackend::InMemory | StorageBackend::Sqlite { .. } => {
                bail!("storage.postgres is only valid when storage.backend=postgres")
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Worker pool
// ─────────────────────────────────────────────────────────────────────

/// Worker pool and lease management settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerConfig {
    /// Number of concurrent workers polling for runnable tasks.
    pub pool_size: usize,
    /// Seconds before an acquired lease expires without a heartbeat.
    pub lease_duration_secs: u64,
    /// Seconds between heartbeat refreshes on an active lease.
    pub heartbeat_interval_secs: u64,
    /// Seconds between expired-lease sweep runs.
    pub sweep_interval_secs: u64,
    /// Seconds between idle worker acquisition polls.
    ///
    /// When a worker has no active task it sleeps for this duration
    /// before calling `acquire_next_runnable` again.  A shorter
    /// interval reduces latency to first pick-up but increases
    /// polling load on the store.
    pub acquisition_interval_secs: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            pool_size: 4,
            lease_duration_secs: 30,
            heartbeat_interval_secs: 10,
            sweep_interval_secs: 5,
            acquisition_interval_secs: 1,
        }
    }
}

impl WorkerConfig {
    /// Lease duration as a [`time::Duration`].
    #[must_use]
    pub fn lease_duration(&self) -> time::Duration {
        let secs = i64::try_from(self.lease_duration_secs).unwrap_or(i64::MAX);
        time::Duration::seconds(secs)
    }

    /// Heartbeat interval as a [`std::time::Duration`] (for tokio timers).
    #[must_use]
    pub const fn heartbeat_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.heartbeat_interval_secs)
    }

    /// Sweep interval as a [`std::time::Duration`] (for tokio timers).
    #[must_use]
    pub const fn sweep_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.sweep_interval_secs)
    }

    /// Acquisition poll interval as a [`std::time::Duration`] (for tokio timers).
    #[must_use]
    pub const fn acquisition_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.acquisition_interval_secs)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Transport
// ─────────────────────────────────────────────────────────────────────

/// Transport enablement flags and bind addresses.
///
/// Transports are disabled by default.  A future gRPC or HTTP crate
/// reads these flags to decide whether to start its listener.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TransportConfig {
    /// Enable the gRPC transport.
    pub grpc_enabled: bool,
    /// Bind address for the gRPC listener.
    pub grpc_addr: SocketAddr,
    /// Enable the HTTP/REST transport.
    pub http_enabled: bool,
    /// Bind address for the HTTP listener.
    pub http_addr: SocketAddr,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            grpc_enabled: false,
            grpc_addr: ([127, 0, 0, 1], 50051).into(),
            http_enabled: false,
            http_addr: ([127, 0, 0, 1], 8080).into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Retention
// ─────────────────────────────────────────────────────────────────────

/// Data retention policies.
///
/// `None` values mean "keep forever" — the sweep tasks skip the
/// corresponding cleanup.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RetentionConfig {
    /// Time-to-live for committed events, in seconds.  `None` = keep
    /// forever.
    pub event_ttl_secs: Option<u64>,
    /// Maximum checkpoints per thread.  `None` = no limit.
    pub checkpoint_max_per_thread: Option<u32>,
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_round_trips_through_yaml() -> Result<()> {
        let original = ServiceConfig::default();
        let yaml = serde_yaml::to_string(&original)?;
        let recovered = ServiceConfig::from_yaml_str(&yaml)?;

        // Spot-check key defaults survived the round trip.
        assert_eq!(recovered.worker.pool_size, 4);
        assert_eq!(recovered.worker.lease_duration_secs, 30);
        assert!(!recovered.transport.grpc_enabled);
        assert!(!recovered.transport.http_enabled);
        assert!(recovered.retention.event_ttl_secs.is_none());
        Ok(())
    }

    #[test]
    fn minimal_yaml_loads_defaults() -> Result<()> {
        let config = ServiceConfig::from_yaml_str("{}")?;
        assert_eq!(config.worker.pool_size, 4);
        assert_eq!(config.worker.sweep_interval_secs, 5);
        Ok(())
    }

    #[test]
    fn partial_yaml_merges_with_defaults() -> Result<()> {
        let yaml = r"
worker:
  pool_size: 16
transport:
  grpc_enabled: true
";
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert_eq!(config.worker.pool_size, 16);
        // Unset fields fall back to defaults.
        assert_eq!(config.worker.lease_duration_secs, 30);
        assert!(config.transport.grpc_enabled);
        assert!(!config.transport.http_enabled);
        Ok(())
    }

    #[test]
    fn storage_backend_in_memory_is_default() {
        let config = ServiceConfig::default();
        assert!(matches!(config.storage.backend, StorageBackend::InMemory));
    }

    #[test]
    fn postgres_storage_config_redacts_database_url_in_debug() {
        let config = PostgresStorageConfig {
            database_url: Some("postgres://secret-user:secret-pass@example.com/db".into()),
            schema: Some("agent_service_host".into()),
            max_connections: 12,
        };

        let rendered = format!("{config:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("secret-pass"));
    }

    #[test]
    fn postgres_backend_yaml_parses() -> Result<()> {
        let yaml = r"
storage:
  backend: postgres
  postgres:
    database_url: postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk
    schema: host_tests
    max_connections: 16
";

        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert!(matches!(config.storage.backend, StorageBackend::Postgres));
        assert_eq!(
            config.storage.postgres.database_url.as_deref(),
            Some("postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk"),
        );
        assert_eq!(
            config.storage.postgres.schema.as_deref(),
            Some("host_tests")
        );
        assert_eq!(config.storage.postgres.max_connections, 16);
        Ok(())
    }

    #[test]
    fn worker_config_duration_helpers() {
        let wc = WorkerConfig::default();
        assert_eq!(wc.lease_duration(), time::Duration::seconds(30));
        assert_eq!(wc.heartbeat_interval(), std::time::Duration::from_secs(10));
        assert_eq!(wc.sweep_interval(), std::time::Duration::from_secs(5));
        assert_eq!(wc.acquisition_interval(), std::time::Duration::from_secs(1));
    }

    #[test]
    fn full_yaml_round_trip() -> Result<()> {
        let yaml = r#"
storage:
  backend: in_memory
worker:
  pool_size: 8
  lease_duration_secs: 60
  heartbeat_interval_secs: 20
  sweep_interval_secs: 10
  acquisition_interval_secs: 2
transport:
  grpc_enabled: true
  grpc_addr: "0.0.0.0:50051"
  http_enabled: true
  http_addr: "0.0.0.0:8080"
retention:
  event_ttl_secs: 86400
  checkpoint_max_per_thread: 100
"#;
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert_eq!(config.worker.pool_size, 8);
        assert_eq!(config.worker.lease_duration_secs, 60);
        assert_eq!(config.worker.acquisition_interval_secs, 2);
        assert!(config.transport.grpc_enabled);
        assert!(config.transport.http_enabled);
        assert_eq!(config.retention.event_ttl_secs, Some(86400));
        assert_eq!(config.retention.checkpoint_max_per_thread, Some(100));

        // Re-serialize and re-parse to prove stability.
        let re_yaml = serde_yaml::to_string(&config)?;
        let re_config = ServiceConfig::from_yaml_str(&re_yaml)?;
        assert_eq!(re_config.worker.pool_size, 8);
        Ok(())
    }

    #[test]
    fn sqlite_backend_minimal_yaml() -> Result<()> {
        // serde_yaml 0.9 uses YAML tags for struct enum variants.
        let yaml = r"
storage:
  backend: !sqlite
    path: null
";
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert!(matches!(
            config.storage.backend,
            StorageBackend::Sqlite { path: None }
        ));
        Ok(())
    }

    #[test]
    fn sqlite_backend_explicit_path_yaml() -> Result<()> {
        let yaml = r#"
storage:
  backend: !sqlite
    path: "/tmp/test.db"
"#;
        let config = ServiceConfig::from_yaml_str(yaml)?;
        match &config.storage.backend {
            StorageBackend::Sqlite { path } => {
                assert_eq!(path.as_deref(), Some("/tmp/test.db"));
            }
            other => panic!("expected Sqlite, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn sqlite_backend_round_trips_through_yaml() -> Result<()> {
        let yaml = r#"
storage:
  backend: !sqlite
    path: "/data/agent-sdk.db"
"#;
        let config = ServiceConfig::from_yaml_str(yaml)?;
        let re_yaml = serde_yaml::to_string(&config)?;
        let re_config = ServiceConfig::from_yaml_str(&re_yaml)?;
        match &re_config.storage.backend {
            StorageBackend::Sqlite { path } => {
                assert_eq!(path.as_deref(), Some("/data/agent-sdk.db"));
            }
            other => panic!("expected Sqlite, got {other:?}"),
        }
        Ok(())
    }
}
