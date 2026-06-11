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
//!   janitor_enabled: false
//!   janitor_interval_secs: 60
//!   janitor_batch_size: 100
//! ```
//!
//! The built-in retention janitor is opt-in: setting `event_ttl_secs`
//! or `checkpoint_max_per_thread` alone is not enough — the janitor
//! only sweeps when `janitor_enabled: true`.  When enabled, the host
//! spawns an extra background task (see the `host` module docs).
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
use std::path::{Path, PathBuf};

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
    /// Outbox relay + broker configuration.
    pub relay: RelayConfig,
    /// Task-wakeup consumer + fallback sweep configuration.
    pub wakeup: crate::wakeup::WakeupConfig,
    /// Cross-instance thread-event watch consumer configuration.
    pub watch: crate::watch::ThreadEventsWatchConfig,
    /// `OpenTelemetry` tracer + meter wiring.
    ///
    /// All fields default to "disabled" so a host built without
    /// `--features otel` (or one started without an `[observability]`
    /// section in YAML) behaves exactly like prior versions.
    pub observability: ObservabilityConfig,
    /// gRPC admission back-pressure and input-size limits.
    pub admission: AdmissionConfig,
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
    /// Embedded `SQLite` database in WAL mode.  Designed for desktop and
    /// CLI processes that own a single data directory.  State survives
    /// process restarts.
    ///
    /// Gated behind the `sqlite` cargo feature.
    Sqlite {
        /// Path to the `SQLite` database file.
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

    /// Return the `SQLite` database URL when the backend is selected.
    ///
    /// When no explicit path is provided, returns a platform-default
    /// data directory path.
    ///
    /// # Errors
    /// Returns an error if the backend is not `sqlite`.
    pub fn sqlite_database_url(&self) -> Result<String> {
        match &self.backend {
            StorageBackend::Sqlite { path } => {
                let db_path = if let Some(path) = path {
                    std::path::PathBuf::from(path)
                } else {
                    dirs_default_sqlite_dir()?.join("agent-sdk.db")
                };
                if let Some(parent) = db_path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create sqlite data dir {}", parent.display()))?;
                }
                Ok(sqlite_url(&db_path))
            }
            _ => bail!("sqlite_database_url is only valid when storage.backend=sqlite"),
        }
    }
}

/// Build a `sqlite://` connection URL for the given filesystem path.
///
/// Two structural pitfalls must be avoided:
/// 1. On Windows, `PathBuf::display()` emits backslashes that sqlx's URL
///    parser rejects — substitute forward slashes.
/// 2. Per RFC 3986, the `//` after the scheme introduces an authority
///    component that ends at the next `/`.  For Unix absolute paths
///    starting with `/`, `sqlite://` + `/tmp/foo` naturally produces
///    `sqlite:///tmp/foo` (empty authority).  For Windows drive-letter
///    paths like `C:/Users/...`, `sqlite://` + `C:/...` would parse
///    `C` as the host and silently drop the drive letter; we therefore
///    prepend an extra `/` so the URL becomes `sqlite:///C:/...`.
///
/// Relative paths use the opaque `sqlite:` form (no `//`) so the path
/// does not get mistaken for an authority.
fn sqlite_url(path: &std::path::Path) -> String {
    let mut rendered = path.display().to_string();
    if std::path::MAIN_SEPARATOR != '/' {
        rendered = rendered.replace(std::path::MAIN_SEPARATOR, "/");
    }
    sqlite_url_from_rendered(&rendered, path.is_absolute())
}

/// Format the connection URL for a path that has already been rendered
/// as a `/`-separated string.  The `is_absolute` flag is supplied by the
/// caller because `Path::is_absolute` is platform-dependent — on Linux
/// it returns `false` for a Windows drive-letter path like
/// `C:/Users/...`, which makes the Windows branch untestable when the
/// decision is made inside this helper.  Factoring the flag out lets
/// unit tests exercise the Windows branch regardless of host OS.
fn sqlite_url_from_rendered(rendered: &str, is_absolute: bool) -> String {
    if is_absolute {
        let normalised = if rendered.starts_with('/') {
            rendered.to_owned()
        } else {
            // Windows drive-letter form (e.g. `C:/Users/...`) — the
            // leading `/` gives us an empty authority so the drive
            // letter survives in the URL path.
            format!("/{rendered}")
        };
        format!("sqlite://{normalised}?mode=rwc")
    } else {
        format!("sqlite:{rendered}?mode=rwc")
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
// Admission back-pressure + input limits
// ─────────────────────────────────────────────────────────────────────

/// gRPC admission back-pressure and input-size limits.
///
/// These guard the `SubmitThreadWork` path against unbounded backlog
/// growth (a retry storm enqueuing serial work) and oversized inputs
/// (large inline base64 attachments). Defaults are generous but finite
/// so a misbehaving client gets a clean `RESOURCE_EXHAUSTED` /
/// `INVALID_ARGUMENT` rather than unbounded memory growth or a transport
/// failure.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdmissionConfig {
    /// Maximum number of *queued* root turns a single thread may hold
    /// (the active/blocking root is not counted). A submission that
    /// would exceed this returns `RESOURCE_EXHAUSTED`. `None` disables
    /// the cap.
    pub max_queued_roots_per_thread: Option<u32>,
    /// Maximum total encoded size, in bytes, of a single
    /// `SubmitThreadWork` request's input items. A request larger than
    /// this returns `INVALID_ARGUMENT`. `None` disables the aggregate
    /// limit.
    pub max_submit_input_bytes: Option<usize>,
    /// Maximum encoded size, in bytes, of any single input item within
    /// a `SubmitThreadWork` request. `None` disables the per-item limit.
    pub max_submit_item_bytes: Option<usize>,
    /// Maximum gRPC decoded message size, in bytes, applied to the
    /// control service via tonic's `max_decoding_message_size`. Bounds
    /// the transport-level buffer so an oversized frame is rejected
    /// before it is fully decoded into memory.
    pub max_decoding_message_bytes: usize,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            // 1024 queued roots per thread is far above any sane backlog
            // but still bounds a retry storm.
            max_queued_roots_per_thread: Some(1024),
            // 8 MiB aggregate / 4 MiB per item keeps large multimodal
            // submissions working while bounding inline-attachment abuse.
            max_submit_input_bytes: Some(8 * 1024 * 1024),
            max_submit_item_bytes: Some(4 * 1024 * 1024),
            // 16 MiB transport ceiling — comfortably above the aggregate
            // input limit so the application-level check (which returns a
            // clean INVALID_ARGUMENT) fires first for normal payloads.
            max_decoding_message_bytes: 16 * 1024 * 1024,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Relay (AMQP outbox relay)
// ─────────────────────────────────────────────────────────────────────

/// Outbox relay and broker configuration.
///
/// When `enabled` is `false` (the default) the service host does not
/// spawn a relay — the outbox still receives rows from journal
/// commits, but no worker publishes them.  This keeps the host trivial
/// to run in environments where the broker is not yet provisioned
/// (local development, CI) while keeping the configuration surface
/// ready for deploys to flip the switch.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RelayConfig {
    /// Whether the relay scheduler runs.
    pub enabled: bool,
    /// Stable worker identifier recorded on every claim.
    ///
    /// `None` means "generate a fresh one at startup" — suitable for
    /// single-process deploys.  Multi-instance deploys should pin a
    /// hostname or pod identifier here so claim reclaims can attribute
    /// stuck rows to a specific crashed worker.
    pub worker_id: Option<String>,
    /// Maximum rows claimed per relay tick.
    pub batch_size: u32,
    /// Seconds between steady-state relay ticks when there is no backlog.
    pub poll_interval_secs: u64,
    /// Seconds a claim is considered valid before a reclaim sweep
    /// treats it as abandoned by a crashed worker.
    pub claim_lease_secs: u64,
    /// Seconds between periodic claim-reclaim sweeps.
    pub reclaim_interval_secs: u64,
    /// Seconds to wait before retrying a failed publish.
    pub retry_backoff_secs: u64,
    /// Soft / hard backlog alerting bands.  When set, the relay
    /// scheduler observes the unpublished outbox count after each
    /// tick and flips the latency layer to `Degraded` if the soft
    /// band is exceeded.  `None` keeps the historical "best-effort"
    /// behaviour with no in-process protection — operators alert
    /// purely from external metrics.
    pub backlog_threshold: Option<crate::metrics::BacklogThreshold>,
    /// Which broker adapter to compose into the relay.
    pub broker: BrokerConfig,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            worker_id: None,
            batch_size: 128,
            poll_interval_secs: 2,
            claim_lease_secs: 60,
            reclaim_interval_secs: 30,
            retry_backoff_secs: 30,
            backlog_threshold: None,
            broker: BrokerConfig::default(),
        }
    }
}

impl RelayConfig {
    /// Steady-state poll interval as a [`std::time::Duration`].
    #[must_use]
    pub const fn poll_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.poll_interval_secs)
    }

    /// Reclaim sweep interval as a [`std::time::Duration`].
    #[must_use]
    pub const fn reclaim_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.reclaim_interval_secs)
    }

    /// Claim-lease duration as a [`time::Duration`].
    #[must_use]
    pub fn claim_lease(&self) -> time::Duration {
        let secs = i64::try_from(self.claim_lease_secs).unwrap_or(i64::MAX);
        time::Duration::seconds(secs)
    }
}

/// Broker-adapter selection.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerConfig {
    /// In-process broker double.  Messages are dropped on the floor
    /// after being recorded by the in-memory adapter — useful for
    /// running the host without a real broker in tests or local dev.
    #[default]
    InMemory,
    /// AMQP 0.9.1 broker (typically `RabbitMQ`).
    ///
    /// Requires the `amqp` feature.
    #[cfg(feature = "amqp")]
    Amqp(crate::broker::amqp::AmqpBrokerConfig),
}

// ─────────────────────────────────────────────────────────────────────
// Retention
// ─────────────────────────────────────────────────────────────────────

/// Data retention policies.
///
/// `None` values mean "keep forever" — the sweep tasks skip the
/// corresponding cleanup.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RetentionConfig {
    /// Time-to-live for committed events, in seconds.  `None` = keep
    /// forever.
    pub event_ttl_secs: Option<u64>,
    /// Maximum checkpoints per thread.  `None` = no limit.
    pub checkpoint_max_per_thread: Option<u32>,
    /// Whether the retention janitor background loop runs.
    pub janitor_enabled: bool,
    /// Seconds between janitor sweep cycles.
    pub janitor_interval_secs: u64,
    /// Maximum threads processed per janitor cycle.
    pub janitor_batch_size: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            event_ttl_secs: None,
            checkpoint_max_per_thread: None,
            janitor_enabled: false,
            janitor_interval_secs: 60,
            janitor_batch_size: 100,
        }
    }
}

impl RetentionConfig {
    /// Janitor sweep interval as a [`std::time::Duration`] (for tokio timers).
    #[must_use]
    pub const fn janitor_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.janitor_interval_secs)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Observability
// ─────────────────────────────────────────────────────────────────────

/// `OpenTelemetry` configuration for the host binary.
///
/// `enabled = false` (the default) is a hard skip — no exporter is
/// installed, no global providers are touched, and the rest of the
/// host runs exactly as it did before this section existed.
///
/// When `enabled = true`, the host calls
/// [`agent_sdk_otel::install_global_provider`](https://docs.rs/agent-sdk-otel)
/// at startup with the values resolved from this section.  Anything
/// left unset on the YAML side falls through to the standard
/// `OTEL_*` environment variables (handled by
/// `OtelConfig::from_env`); explicit YAML values *override* the env
/// values.  This keeps containerised deploys 12-factor while still
/// allowing static config files to pin values.
///
/// `service_name` defaults to `agent-service-host` so dashboards
/// have a useful label even when nothing is configured.
///
/// # Example
///
/// ```yaml
/// observability:
///   enabled: true
///   otlp_endpoint: "http://localhost:4317"
///   service_name: agent-service-host
///   deployment_environment: local
///   sampler: parentbased_traceidratio
///   sample_ratio: 1.0
/// ```
///
/// `Debug` is implemented by hand (not derived) so the secret-bearing
/// `otlp_headers` field is rendered as a count, never as its values —
/// the whole `ServiceConfig` is `Debug`-logged at startup
/// (`main.rs`), so a derived impl would leak vendor auth tokens into
/// the logs.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ObservabilityConfig {
    /// Master switch.  When `false`, the host skips every step of the
    /// `OTel` install path and behaves like a build without
    /// `--features otel`.
    pub enabled: bool,
    /// `service.name` resource attribute.  Defaults to
    /// `agent-service-host` when unset; YAML wins over `OTEL_SERVICE_NAME`.
    pub service_name: Option<String>,
    /// `service.instance.id` resource attribute.  Defaults to a
    /// per-process UUID generated by `agent-sdk-otel` when unset.
    pub service_instance_id: Option<String>,
    /// `deployment.environment` resource attribute.
    pub deployment_environment: Option<String>,
    /// OTLP gRPC endpoint.  An explicitly empty string disables the
    /// exporter even when `enabled = true` — useful for smoke-testing
    /// the install path without a collector running.
    pub otlp_endpoint: Option<String>,
    /// OTLP gRPC headers, typically used for vendor auth.  Carries
    /// secrets in production, so the field is included in `Debug` only
    /// as a count (the underlying `OtelConfig::Debug` impl masks
    /// values).
    #[serde(default)]
    pub otlp_headers: Vec<(String, String)>,
    /// Sampler name (matches `OTEL_TRACES_SAMPLER`).  Defaults to
    /// `parentbased_traceidratio` with `sample_ratio = 1.0` so local
    /// dev "just works".
    pub sampler: Option<String>,
    /// Ratio for ratio-based samplers in `[0.0, 1.0]`.  Ignored
    /// otherwise.
    pub sample_ratio: Option<f64>,
    /// Baggage keys allowed to leave the process via the W3C baggage
    /// propagator.  Empty falls back to
    /// `AllowListBaggagePropagator::baseline_allow_list()`.
    #[serde(default)]
    pub propagated_baggage_keys: Vec<String>,
    /// Whether `gen_ai.input.messages` / `gen_ai.output.messages` may
    /// be captured as span attributes.  Default-deny.
    pub capture_payloads: bool,
}

impl std::fmt::Debug for ObservabilityConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservabilityConfig")
            .field("enabled", &self.enabled)
            .field("service_name", &self.service_name)
            .field("service_instance_id", &self.service_instance_id)
            .field("deployment_environment", &self.deployment_environment)
            .field("otlp_endpoint", &self.otlp_endpoint)
            // Secret-bearing: render only the count, never the values.
            .field("otlp_headers", &self.otlp_headers.len())
            .field("sampler", &self.sampler)
            .field("sample_ratio", &self.sample_ratio)
            .field("propagated_baggage_keys", &self.propagated_baggage_keys)
            .field("capture_payloads", &self.capture_payloads)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Platform-default SQLite data directory
// ─────────────────────────────────────────────────────────────────────

/// Return the platform-default data directory for `SQLite`.
///
/// - Linux:   `$XDG_DATA_HOME/agent-sdk` (defaults to `~/.local/share/agent-sdk`)
/// - macOS:   `~/Library/Application Support/agent-sdk`
/// - Windows: `%LOCALAPPDATA%\agent-sdk`
fn dirs_default_sqlite_dir() -> Result<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME").context("HOME not set")?;
        Ok(PathBuf::from(home).join("Library/Application Support/agent-sdk"))
    }
    #[cfg(target_os = "linux")]
    {
        let base = std::env::var("XDG_DATA_HOME")
            .or_else(|_| std::env::var("HOME").map(|home| format!("{home}/.local/share")))
            .context("neither XDG_DATA_HOME nor HOME is set")?;
        Ok(PathBuf::from(base).join("agent-sdk"))
    }
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var("LOCALAPPDATA").context("LOCALAPPDATA not set")?;
        Ok(PathBuf::from(base).join("agent-sdk"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        anyhow::bail!("unsupported platform for default SQLite data directory")
    }
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

    #[test]
    fn sqlite_url_unix_absolute_uses_three_slashes() {
        let url = sqlite_url(std::path::Path::new("/var/lib/agent-sdk.db"));
        assert_eq!(url, "sqlite:///var/lib/agent-sdk.db?mode=rwc");
    }

    #[test]
    fn sqlite_url_windows_drive_letter_keeps_drive() {
        // Drive-letter paths are not absolute on Linux, so call the
        // inner helper directly with is_absolute=true (as Windows
        // would report) to exercise the real URL-formation logic.
        let url = sqlite_url_from_rendered("C:/Users/me/agent-sdk.db", true);
        assert_eq!(url, "sqlite:///C:/Users/me/agent-sdk.db?mode=rwc");
    }

    #[test]
    fn sqlite_url_from_rendered_unix_absolute() {
        let url = sqlite_url_from_rendered("/var/lib/agent-sdk.db", true);
        assert_eq!(url, "sqlite:///var/lib/agent-sdk.db?mode=rwc");
    }

    #[test]
    fn sqlite_url_from_rendered_relative_uses_opaque_form() {
        let url = sqlite_url_from_rendered("local.db", false);
        assert_eq!(url, "sqlite:local.db?mode=rwc");
    }

    #[test]
    fn sqlite_url_relative_uses_opaque_form() {
        let url = sqlite_url(std::path::Path::new("local.db"));
        assert_eq!(url, "sqlite:local.db?mode=rwc");
    }

    #[test]
    fn relay_defaults_are_disabled_with_in_memory_broker() {
        let config = RelayConfig::default();
        assert!(!config.enabled);
        assert!(matches!(config.broker, BrokerConfig::InMemory));
        assert_eq!(config.batch_size, 128);
        assert_eq!(config.poll_interval_secs, 2);
        assert_eq!(config.reclaim_interval_secs, 30);
        assert_eq!(config.claim_lease_secs, 60);
        assert!(
            config.backlog_threshold.is_none(),
            "backlog protection is opt-in",
        );
    }

    #[test]
    fn relay_duration_helpers() {
        let config = RelayConfig::default();
        assert_eq!(config.poll_interval(), std::time::Duration::from_secs(2));
        assert_eq!(
            config.reclaim_interval(),
            std::time::Duration::from_secs(30)
        );
        assert_eq!(config.claim_lease(), time::Duration::seconds(60));
    }

    #[test]
    fn relay_yaml_in_memory_broker_round_trips() -> Result<()> {
        let yaml = r"
relay:
  enabled: true
  batch_size: 32
  poll_interval_secs: 5
  claim_lease_secs: 45
  broker: in_memory
";
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert!(config.relay.enabled);
        assert_eq!(config.relay.batch_size, 32);
        assert_eq!(config.relay.poll_interval_secs, 5);
        assert_eq!(config.relay.claim_lease_secs, 45);
        assert!(config.relay.backlog_threshold.is_none());
        assert!(matches!(config.relay.broker, BrokerConfig::InMemory));
        Ok(())
    }

    #[test]
    fn relay_yaml_backlog_threshold_round_trips() -> Result<()> {
        let yaml = r"
relay:
  enabled: true
  backlog_threshold:
    soft: 250
    hard: 5000
  broker: in_memory
";
        let config = ServiceConfig::from_yaml_str(yaml)?;
        let threshold = config.relay.backlog_threshold.context("threshold parsed")?;
        assert_eq!(threshold.soft, 250);
        assert_eq!(threshold.hard, 5_000);

        let re_yaml = serde_yaml::to_string(&config)?;
        let re_config = ServiceConfig::from_yaml_str(&re_yaml)?;
        let recovered = re_config
            .relay
            .backlog_threshold
            .context("threshold survives round trip")?;
        assert_eq!(recovered.soft, 250);
        assert_eq!(recovered.hard, 5_000);
        Ok(())
    }

    #[cfg(feature = "amqp")]
    #[test]
    fn relay_yaml_amqp_broker_parses() -> Result<()> {
        let yaml = r#"
relay:
  enabled: true
  broker: !amqp
    url: "amqp://user:pass@broker.internal:5672/prod"
    exchange: "agent_sdk.outbox"
    exchange_kind: topic
    declare_exchange: false
    routing_key_prefix: "agent_sdk.outbox"
"#;
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert!(config.relay.enabled);
        match config.relay.broker {
            BrokerConfig::Amqp(amqp) => {
                assert_eq!(
                    amqp.url.as_deref(),
                    Some("amqp://user:pass@broker.internal:5672/prod")
                );
                assert_eq!(amqp.exchange, "agent_sdk.outbox");
                assert!(!amqp.declare_exchange);
            }
            BrokerConfig::InMemory => panic!("expected AMQP broker"),
        }
        Ok(())
    }

    #[test]
    fn watch_defaults_are_disabled() {
        // Cross-instance watch fanout is off by default so a host
        // that has not provisioned broker queues falls back to
        // journal-only replay (clients reconnect → durable replay).
        let config = ServiceConfig::default();
        assert!(!config.watch.enabled);
    }

    #[cfg(feature = "amqp")]
    #[test]
    fn watch_yaml_with_amqp_consumer_parses() -> Result<()> {
        let yaml = r"
watch:
  enabled: true
  amqp_consumer:
    enabled: true
    config:
      queue: 'agent_sdk.thread_events.pod-x'
      consumer_tag_prefix: 'pod-x-watch'
      declare_queue: true
      bind_queue: true
";
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert!(config.watch.enabled);
        assert!(config.watch.amqp_consumer.enabled);
        assert_eq!(
            config.watch.amqp_consumer.config.queue,
            "agent_sdk.thread_events.pod-x",
        );
        assert!(config.watch.amqp_consumer.config.declare_queue);
        assert!(config.watch.amqp_consumer.config.bind_queue);

        // Round-trip the parsed config through YAML to prove the
        // serializer produces input the parser accepts.
        let re_yaml = serde_yaml::to_string(&config)?;
        let re_config = ServiceConfig::from_yaml_str(&re_yaml)?;
        assert!(re_config.watch.enabled);
        assert!(re_config.watch.amqp_consumer.enabled);
        assert_eq!(
            re_config.watch.amqp_consumer.config.queue,
            "agent_sdk.thread_events.pod-x",
        );
        assert!(re_config.watch.amqp_consumer.config.declare_queue);
        assert!(re_config.watch.amqp_consumer.config.bind_queue);
        Ok(())
    }

    // ── observability ───────────────────────────────────────────────

    // ── admission back-pressure + input limits ──────────────────────

    #[test]
    fn admission_defaults_are_finite_but_generous() {
        let config = AdmissionConfig::default();
        assert_eq!(config.max_queued_roots_per_thread, Some(1024));
        assert_eq!(config.max_submit_input_bytes, Some(8 * 1024 * 1024));
        assert_eq!(config.max_submit_item_bytes, Some(4 * 1024 * 1024));
        assert_eq!(config.max_decoding_message_bytes, 16 * 1024 * 1024);
    }

    #[test]
    fn admission_yaml_round_trips() -> Result<()> {
        let yaml = r"
admission:
  max_queued_roots_per_thread: 8
  max_submit_input_bytes: 2048
  max_submit_item_bytes: 1024
  max_decoding_message_bytes: 65536
";
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert_eq!(config.admission.max_queued_roots_per_thread, Some(8));
        assert_eq!(config.admission.max_submit_input_bytes, Some(2048));
        assert_eq!(config.admission.max_submit_item_bytes, Some(1024));
        assert_eq!(config.admission.max_decoding_message_bytes, 65536);

        let re_yaml = serde_yaml::to_string(&config)?;
        let re_config = ServiceConfig::from_yaml_str(&re_yaml)?;
        assert_eq!(re_config.admission.max_queued_roots_per_thread, Some(8));
        Ok(())
    }

    #[test]
    fn admission_null_disables_caps() -> Result<()> {
        let yaml = r"
admission:
  max_queued_roots_per_thread: null
  max_submit_input_bytes: null
  max_submit_item_bytes: null
";
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert!(config.admission.max_queued_roots_per_thread.is_none());
        assert!(config.admission.max_submit_input_bytes.is_none());
        assert!(config.admission.max_submit_item_bytes.is_none());
        // Unset transport ceiling falls back to the default.
        assert_eq!(
            config.admission.max_decoding_message_bytes,
            16 * 1024 * 1024
        );
        Ok(())
    }

    #[test]
    fn full_service_config_debug_redacts_secrets() {
        // The whole ServiceConfig is Debug-logged at startup
        // (`main.rs`). A fully-populated config must never surface AMQP
        // credentials or OTLP auth-header values in its Debug output.
        let mut config = ServiceConfig::default();
        config.observability.otlp_headers = vec![
            (
                "authorization".to_string(),
                "Bearer super-secret-token".to_string(),
            ),
            ("x-api-key".to_string(), "another-secret-value".to_string()),
        ];

        #[cfg(feature = "amqp")]
        {
            config.relay.broker = BrokerConfig::Amqp(crate::broker::amqp::AmqpBrokerConfig {
                url: Some("amqp://rabbit:hunter2@broker.internal:5672/prod".into()),
                ..crate::broker::amqp::AmqpBrokerConfig::default()
            });
            config.wakeup.amqp_consumer.config.broker.url =
                Some("amqp://wakeup:wakeup-pass@broker.internal:5672/prod".into());
            config.watch.amqp_consumer.config.broker.url =
                Some("amqp://watch:watch-pass@broker.internal:5672/prod".into());
        }

        let rendered = format!("{config:?}");
        for secret in [
            "super-secret-token",
            "another-secret-value",
            "hunter2",
            "wakeup-pass",
            "watch-pass",
        ] {
            assert!(
                !rendered.contains(secret),
                "secret '{secret}' leaked in ServiceConfig Debug output: {rendered}"
            );
        }
        // The redacted header count is still observable for diagnostics.
        assert!(rendered.contains("otlp_headers: 2"));
    }

    #[test]
    fn observability_defaults_are_disabled() {
        // Hard skip: a clean default config never installs any OTel
        // pipeline.  This is what every existing host deployment
        // gets after upgrading, so behaviour stays stable.
        let config = ServiceConfig::default();
        assert!(!config.observability.enabled);
        assert!(config.observability.service_name.is_none());
        assert!(config.observability.otlp_endpoint.is_none());
        assert!(config.observability.otlp_headers.is_empty());
        assert!(config.observability.sampler.is_none());
        assert!(config.observability.sample_ratio.is_none());
        assert!(config.observability.propagated_baggage_keys.is_empty());
        assert!(!config.observability.capture_payloads);
    }

    #[test]
    fn observability_yaml_round_trip() -> Result<()> {
        let yaml = r#"
observability:
  enabled: true
  service_name: agent-service-host
  service_instance_id: host-01
  deployment_environment: local
  otlp_endpoint: "http://localhost:4317"
  otlp_headers:
    - ["authorization", "Bearer secret"]
  sampler: parentbased_traceidratio
  sample_ratio: 0.25
  propagated_baggage_keys: ["user.id", "session.id"]
  capture_payloads: true
"#;
        let config = ServiceConfig::from_yaml_str(yaml)?;
        assert!(config.observability.enabled);
        assert_eq!(
            config.observability.service_name.as_deref(),
            Some("agent-service-host"),
        );
        assert_eq!(
            config.observability.service_instance_id.as_deref(),
            Some("host-01"),
        );
        assert_eq!(
            config.observability.deployment_environment.as_deref(),
            Some("local"),
        );
        assert_eq!(
            config.observability.otlp_endpoint.as_deref(),
            Some("http://localhost:4317"),
        );
        assert_eq!(config.observability.otlp_headers.len(), 1);
        assert_eq!(
            config.observability.sampler.as_deref(),
            Some("parentbased_traceidratio"),
        );
        assert!(
            config
                .observability
                .sample_ratio
                .is_some_and(|ratio| (ratio - 0.25).abs() < f64::EPSILON),
        );
        assert_eq!(
            config.observability.propagated_baggage_keys,
            vec!["user.id".to_string(), "session.id".to_string()],
        );
        assert!(config.observability.capture_payloads);

        let re_yaml = serde_yaml::to_string(&config)?;
        let re_config = ServiceConfig::from_yaml_str(&re_yaml)?;
        assert!(re_config.observability.enabled);
        assert_eq!(
            re_config.observability.otlp_endpoint.as_deref(),
            Some("http://localhost:4317"),
        );
        Ok(())
    }
}
