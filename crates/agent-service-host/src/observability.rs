//! `agent_service_host.*` and `OTel`-spec metric instruments.
//!
//! Compiled only with `feature = "otel"`. Mirrors the lazy-singleton
//! pattern used by `agent_server::observability::ServerMetrics`:
//! instruments are bound to whichever `MeterProvider` is current at
//! the first call to [`HostMetrics::global`] / [`HostMetrics::init`];
//! tests rotate the provider via
//! [`HostMetrics::reset_for_testing`].
//!
//! ## What we record
//!
//! | Metric | Why |
//! |--------|-----|
//! | `rpc.server.duration` (`Histogram`, `s`) | `OTel` RPC semconv. Driven by [`grpc_layer::MetricsLayer`]. |
//! | `db.client.connections.create_time` (`Histogram`, `s`) | `sqlx` pool connection establishment latency. |
//! | `db.pool.connections.active` (`ObservableGauge`, `u64`) | Pool busy connection count. |
//! | `db.pool.connections.idle` (`ObservableGauge`, `u64`) | Pool idle connection count. |
//! | `agent_service_host.amqp.publish.duration` (`Histogram`, `s`) | `AMQP` publish + confirm latency. |
//! | `agent_service_host.amqp.consume.duration` (`Histogram`, `s`) | `AMQP` delivery → ack/nack latency. |
//! | `agent_service_host.amqp.queue.depth` (`ObservableGauge`, `u64`) | Queue backlog from `queue_declare(passive=true)`. |
//!
//! ## Naming
//!
//! `gRPC` and DB metrics use `OTel` semconv names directly so
//! dashboards and alerts match upstream documentation. `AMQP`-side
//! metrics live under the `agent_service_host.amqp.*` namespace
//! because `OTel` does not yet ship a stable `AMQP` semconv at the
//! time of writing.

use std::sync::{Arc, RwLock};

use anyhow::Context as _;
use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::metrics::{Histogram, ObservableGauge};

// ─────────────────────────────────────────────────────────────────────
// Bucket boundaries
// ─────────────────────────────────────────────────────────────────────

/// Bucket boundaries for `rpc.server.duration` and DB / AMQP timing
/// histograms (seconds). 1 ms .. 5 s spans the realistic RPC range
/// without wasting buckets at either tail.
const SHORT_DURATION_BUCKETS_S: &[f64] = &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0];

// ─────────────────────────────────────────────────────────────────────
// Attribute keys
// ─────────────────────────────────────────────────────────────────────

/// Stable string keys + values used as metric attributes.
pub mod attrs {
    pub const RPC_SYSTEM: &str = "rpc.system";
    pub const RPC_SERVICE: &str = "rpc.service";
    pub const RPC_METHOD: &str = "rpc.method";
    pub const RPC_GRPC_STATUS_CODE: &str = "rpc.grpc.status_code";
    pub const RPC_SYSTEM_GRPC: &str = "grpc";

    pub const DB_POOL_NAME: &str = "db.client.connection.pool.name";
    pub const DB_SYSTEM: &str = "db.system";
    pub const DB_SYSTEM_POSTGRESQL: &str = "postgresql";

    pub const AMQP_DESTINATION: &str = "messaging.destination.name";
    pub const AMQP_OPERATION: &str = "messaging.operation";
    pub const AMQP_OPERATION_PUBLISH: &str = "publish";
    pub const AMQP_OPERATION_RECEIVE: &str = "receive";
    pub const AMQP_OUTCOME: &str = "outcome";
    pub const AMQP_OUTCOME_ACK: &str = "ack";
    pub const AMQP_OUTCOME_NACK: &str = "nack";
    pub const AMQP_OUTCOME_OK: &str = "ok";
    pub const AMQP_OUTCOME_ERROR: &str = "error";
}

// ─────────────────────────────────────────────────────────────────────
// HostMetrics singleton
// ─────────────────────────────────────────────────────────────────────

/// Container for every `agent-service-host` `OTel` instrument.
///
/// Held behind an `Arc` so call sites that need frequent recording
/// (per RPC, per AMQP delivery) clone the handle without locks.
pub struct HostMetrics {
    pub(crate) rpc_server_duration: Histogram<f64>,
    pub(crate) db_client_connection_create_time: Histogram<f64>,
    pub(crate) amqp_publish_duration: Histogram<f64>,
    pub(crate) amqp_consume_duration: Histogram<f64>,
    /// Shared meter handle reused by callback-based gauge
    /// registrations so they bind to the same `MeterProvider` as
    /// every other instrument on this struct.
    meter: opentelemetry::metrics::Meter,
}

impl std::fmt::Debug for HostMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostMetrics").finish_non_exhaustive()
    }
}

static METRICS: RwLock<Option<Arc<HostMetrics>>> = RwLock::new(None);

impl HostMetrics {
    /// Build instruments under `name` and cache the resulting handle
    /// for subsequent [`HostMetrics::global`] callers.
    ///
    /// The first caller in the process wins; later calls return the
    /// cached handle and ignore `name`. Tests rotating providers
    /// must call [`HostMetrics::reset_for_testing`] beforehand.
    #[must_use]
    pub fn init(name: &'static str) -> Arc<Self> {
        if let Some(existing) = read_cached() {
            return existing;
        }

        let built = Arc::new(Self::build(name));
        write_cached(Arc::clone(&built));
        built
    }

    /// Convenience wrapper that initialises the singleton under the
    /// `agent-service-host` meter scope.
    #[must_use]
    pub fn global() -> Arc<Self> {
        Self::init(env!("CARGO_PKG_NAME"))
    }

    /// Drop the cached singleton. Test-only.
    pub fn reset_for_testing() {
        let mut guard = match METRICS.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = None;
    }

    /// Record a single `rpc.server.duration` sample.
    ///
    /// `service` and `method` are derived from the HTTP `:path`
    /// header — `/agent.service.v1.AgentControlService/CreateThread`
    /// becomes `service="agent.service.v1.AgentControlService"`,
    /// `method="CreateThread"`. `status_code` is the numeric
    /// `grpc-status` from the response (0 = OK).
    pub fn record_rpc_server_duration(
        &self,
        service: &str,
        method: &str,
        status_code: i32,
        duration_secs: f64,
    ) {
        self.rpc_server_duration.record(
            duration_secs,
            &[
                KeyValue::new(attrs::RPC_SYSTEM, attrs::RPC_SYSTEM_GRPC),
                KeyValue::new(attrs::RPC_SERVICE, service.to_owned()),
                KeyValue::new(attrs::RPC_METHOD, method.to_owned()),
                KeyValue::new(attrs::RPC_GRPC_STATUS_CODE, i64::from(status_code)),
            ],
        );
    }

    /// Record a single `db.client.connections.create_time` sample
    /// taken from the `after_connect` callback.
    pub fn record_db_client_connection_create_time(&self, pool_name: &str, duration_secs: f64) {
        self.db_client_connection_create_time.record(
            duration_secs,
            &[
                KeyValue::new(attrs::DB_SYSTEM, attrs::DB_SYSTEM_POSTGRESQL),
                KeyValue::new(attrs::DB_POOL_NAME, pool_name.to_owned()),
            ],
        );
    }

    /// Record one AMQP publish round-trip.
    pub fn record_amqp_publish(&self, exchange: &str, duration_secs: f64, ok: bool) {
        self.amqp_publish_duration.record(
            duration_secs,
            &[
                KeyValue::new(attrs::AMQP_OPERATION, attrs::AMQP_OPERATION_PUBLISH),
                KeyValue::new(attrs::AMQP_DESTINATION, exchange.to_owned()),
                KeyValue::new(
                    attrs::AMQP_OUTCOME,
                    if ok {
                        attrs::AMQP_OUTCOME_OK
                    } else {
                        attrs::AMQP_OUTCOME_ERROR
                    },
                ),
            ],
        );
    }

    /// Record one AMQP delivery → ack/nack round-trip.
    pub fn record_amqp_consume(&self, queue: &str, duration_secs: f64, ack: bool) {
        self.amqp_consume_duration.record(
            duration_secs,
            &[
                KeyValue::new(attrs::AMQP_OPERATION, attrs::AMQP_OPERATION_RECEIVE),
                KeyValue::new(attrs::AMQP_DESTINATION, queue.to_owned()),
                KeyValue::new(
                    attrs::AMQP_OUTCOME,
                    if ack {
                        attrs::AMQP_OUTCOME_ACK
                    } else {
                        attrs::AMQP_OUTCOME_NACK
                    },
                ),
            ],
        );
    }

    /// Register `(active, idle)` gauges for a sqlx Postgres pool.
    ///
    /// Returns the gauge handles; dropping them deregisters the
    /// callbacks. Caller holds them for the lifetime of the host.
    /// `pool_name` is stamped as the `db.client.connection.pool.name`
    /// attribute.
    #[cfg(feature = "postgres")]
    #[must_use]
    pub fn register_postgres_pool_gauges(
        &self,
        pool_name: &str,
        pool: sqlx::PgPool,
    ) -> (ObservableGauge<u64>, ObservableGauge<u64>) {
        let pool_for_active = pool.clone();
        let pool_name_active = pool_name.to_owned();
        let active = self
            .meter
            .u64_observable_gauge("db.pool.connections.active")
            .with_description("Connections currently checked out from the pool.")
            .with_callback(move |observer| {
                let total = u64::from(pool_for_active.size());
                let Ok(idle) = pool_size_to_u64(pool_for_active.num_idle()) else {
                    tracing::warn!(
                        pool = %pool_name_active,
                        "skipping db.pool.connections.active sample: idle count overflows u64",
                    );
                    return;
                };
                let active = total.saturating_sub(idle);
                observer.observe(
                    active,
                    &[
                        KeyValue::new(attrs::DB_SYSTEM, attrs::DB_SYSTEM_POSTGRESQL),
                        KeyValue::new(attrs::DB_POOL_NAME, pool_name_active.clone()),
                    ],
                );
            })
            .build();

        let pool_for_idle = pool;
        let pool_name_idle = pool_name.to_owned();
        let idle = self
            .meter
            .u64_observable_gauge("db.pool.connections.idle")
            .with_description("Connections currently idle in the pool.")
            .with_callback(move |observer| {
                let Ok(idle) = pool_size_to_u64(pool_for_idle.num_idle()) else {
                    tracing::warn!(
                        pool = %pool_name_idle,
                        "skipping db.pool.connections.idle sample: idle count overflows u64",
                    );
                    return;
                };
                observer.observe(
                    idle,
                    &[
                        KeyValue::new(attrs::DB_SYSTEM, attrs::DB_SYSTEM_POSTGRESQL),
                        KeyValue::new(attrs::DB_POOL_NAME, pool_name_idle.clone()),
                    ],
                );
            })
            .build();

        (active, idle)
    }

    /// Register a callback that publishes
    /// `agent_service_host.amqp.queue.depth` on every metric export.
    ///
    /// Caller is responsible for any rate-limiting on the underlying
    /// query. The `provider` closure is called once per export;
    /// owners typically poll `queue_declare(passive=true)` on a
    /// schedule and stash the latest value in an `Arc<AtomicU64>`.
    #[must_use]
    pub fn register_amqp_queue_depth_gauge(
        &self,
        queue: &str,
        provider: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> ObservableGauge<u64> {
        let queue = queue.to_owned();
        self.meter
            .u64_observable_gauge("agent_service_host.amqp.queue.depth")
            .with_description("AMQP queue depth observed via passive queue_declare.")
            .with_callback(move |observer| {
                observer.observe(
                    provider(),
                    &[KeyValue::new(attrs::AMQP_DESTINATION, queue.clone())],
                );
            })
            .build()
    }

    fn build(scope: &'static str) -> Self {
        let meter = global::meter(scope);

        let rpc_server_duration = meter
            .f64_histogram("rpc.server.duration")
            .with_unit("s")
            .with_description("Duration of inbound gRPC requests.")
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();
        let db_client_connection_create_time = meter
            .f64_histogram("db.client.connections.create_time")
            .with_unit("s")
            .with_description("Time taken to establish a new database connection.")
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();
        let amqp_publish_duration = meter
            .f64_histogram("agent_service_host.amqp.publish.duration")
            .with_unit("s")
            .with_description("AMQP publish + confirm round-trip duration.")
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();
        let amqp_consume_duration = meter
            .f64_histogram("agent_service_host.amqp.consume.duration")
            .with_unit("s")
            .with_description("AMQP delivery → ack/nack duration.")
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();

        Self {
            rpc_server_duration,
            db_client_connection_create_time,
            amqp_publish_duration,
            amqp_consume_duration,
            meter,
        }
    }
}

fn read_cached() -> Option<Arc<HostMetrics>> {
    let guard = match METRICS.read() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.as_ref().map(Arc::clone)
}

fn write_cached(value: Arc<HostMetrics>) {
    let mut guard = match METRICS.write() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.is_none() {
        *guard = Some(value);
    }
}

/// Convert a sqlx pool size from `usize` to `u64` for the metric
/// pipeline.
///
/// Returns `Err` only on hypothetical 128-bit targets where `usize`
/// is wider than `u64`. Callers in the gauge-callback path log on
/// `Err` and skip the metric for that export — the project rule
/// forbids silent fallbacks, including for numeric conversions
/// that are "safe in practice".
#[cfg(feature = "postgres")]
fn pool_size_to_u64(value: usize) -> Result<u64, std::num::TryFromIntError> {
    u64::try_from(value)
}

// ─────────────────────────────────────────────────────────────────────
// Path parsing
// ─────────────────────────────────────────────────────────────────────

/// Split a gRPC HTTP path into `(service, method)`.
///
/// Tonic always serves on paths shaped like
/// `/<rpc.service>/<rpc.method>`. Anything else (`/`, `/foo`, empty)
/// is rejected and the caller should skip the metric — recording
/// with empty attribute values would clobber dashboards.
#[must_use]
pub fn parse_grpc_path(path: &str) -> Option<(&str, &str)> {
    let trimmed = path.strip_prefix('/')?;
    let (service, method) = trimmed.rsplit_once('/')?;
    if service.is_empty() || method.is_empty() {
        return None;
    }
    Some((service, method))
}

// ─────────────────────────────────────────────────────────────────────
// Tower layer for gRPC server-side metrics
// ─────────────────────────────────────────────────────────────────────

pub mod grpc_layer {
    //! Tower middleware that records `rpc.server.duration` for every
    //! gRPC call processed by a tonic server.

    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use std::time::Instant;

    use http::HeaderMap;
    use tower::{Layer, Service};

    use super::{HostMetrics, parse_grpc_path};

    /// Layer that wraps a tonic `Service` with
    /// [`MetricsService`].
    #[derive(Clone, Debug, Default)]
    pub struct MetricsLayer;

    impl MetricsLayer {
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    impl<S> Layer<S> for MetricsLayer {
        type Service = MetricsService<S>;

        fn layer(&self, inner: S) -> Self::Service {
            MetricsService { inner }
        }
    }

    /// `tower::Service` wrapper that captures request method + status
    /// code and records `rpc.server.duration`.
    #[derive(Clone, Debug)]
    pub struct MetricsService<S> {
        inner: S,
    }

    impl<S, ReqBody, ResBody> Service<http::Request<ReqBody>> for MetricsService<S>
    where
        S: Service<http::Request<ReqBody>, Response = http::Response<ResBody>>
            + Clone
            + Send
            + 'static,
        S::Future: Send + 'static,
        ReqBody: Send + 'static,
        ResBody: Send + 'static,
    {
        type Response = S::Response;
        type Error = S::Error;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            self.inner.poll_ready(cx)
        }

        fn call(&mut self, req: http::Request<ReqBody>) -> Self::Future {
            // The `Clone` + `mem::replace` dance is the standard
            // tower pattern for forwarding under a possibly-not-Ready
            // `inner`: we hand the future the ready clone we just
            // poll_readied above.
            let clone = self.inner.clone();
            let mut inner = std::mem::replace(&mut self.inner, clone);

            let path = req.uri().path().to_owned();
            let started_at = Instant::now();

            Box::pin(async move {
                let response = inner.call(req).await?;
                let status_code = grpc_status_from_headers(response.headers());

                if let Some((service, method)) = parse_grpc_path(&path) {
                    HostMetrics::global().record_rpc_server_duration(
                        service,
                        method,
                        status_code,
                        started_at.elapsed().as_secs_f64(),
                    );
                }

                Ok(response)
            })
        }
    }

    /// Read `grpc-status` from response headers if present, else
    /// default to `0` (OK). Trailing-only statuses (the normal happy
    /// path for unary responses) land in the response trailers, not
    /// the headers we see here — the tonic server only writes
    /// `grpc-status` to the head when the call short-circuited with
    /// an error. That asymmetry means the metric undercounts non-OK
    /// codes for streaming RPCs that error mid-stream; B5 accepts
    /// that as a known limitation and the upstream streaming
    /// trailers integration is left for a future card.
    fn grpc_status_from_headers(headers: &HeaderMap) -> i32 {
        // Per the gRPC wire contract the absence of the
        // `grpc-status` header is exactly equivalent to status = OK.
        // Likewise the spec calls for `Unknown` (2) on a malformed
        // status string. Neither branch is a silent fallback — they
        // are the documented spec values.
        let Some(value) = headers.get("grpc-status") else {
            return 0;
        };
        let Ok(text) = value.to_str() else {
            return 2;
        };
        let Ok(code) = text.parse::<i32>() else {
            return 2;
        };
        code
    }
}

// ─────────────────────────────────────────────────────────────────────
// E1: bootstrap helpers (Phase 9)
// ─────────────────────────────────────────────────────────────────────

/// Default `service.name` used when the `[observability]` section
/// does not override it. Aligns with the binary name so dashboards
/// have a useful label even on a clean checkout.
const DEFAULT_SERVICE_NAME: &str = "agent-service-host";

/// Re-export of [`agent_sdk_otel::OtelGuard`] so callers don't need
/// a second `use agent_sdk_otel::OtelGuard;`.
pub use agent_sdk_otel::OtelGuard;

/// Install the global `OTel` tracer + meter providers when the
/// caller's [`ObservabilityConfig::enabled`] is `true`.
///
/// The returned `Option<OtelGuard>` is `None` when
/// `observability.enabled = false`, signalling "skip cleanly". The
/// caller must hold the `Some(guard)` for the lifetime of the
/// process — dropping it flushes pending exports.
///
/// Resolution order for every field:
/// 1. Read the `OTEL_*` environment variables via
///    [`agent_sdk_otel::OtelConfig::from_env`].
/// 2. Override anything the [`ObservabilityConfig`] explicitly sets.
///
/// This keeps containerised deploys 12-factor while still letting
/// static config files pin values when they need to.
///
/// # Errors
/// Returns an error if env parsing fails, the OTLP endpoint is
/// malformed, or the configured sampler is unknown. The host should
/// surface the error and refuse to start rather than running with
/// half-installed observability.
pub fn install_observability(
    cfg: &super::config::ObservabilityConfig,
) -> anyhow::Result<Option<OtelGuard>> {
    if !cfg.enabled {
        return Ok(None);
    }

    let otel_config = build_otel_config(cfg).context("assembling OtelConfig from host config")?;
    let guard = agent_sdk_otel::install_global_provider(&otel_config)
        .context("installing global OTel provider")?;
    Ok(Some(guard))
}

/// Layer the YAML overrides on top of the env-derived `OtelConfig`.
fn build_otel_config(
    cfg: &super::config::ObservabilityConfig,
) -> anyhow::Result<agent_sdk_otel::OtelConfig> {
    use std::str::FromStr;

    use agent_sdk_otel::SamplerKind;

    let mut otel = agent_sdk_otel::OtelConfig::from_env().context("reading OTEL_* env vars")?;

    if let Some(name) = &cfg.service_name {
        otel.service_name.clone_from(name);
    } else if otel.service_name == "agent-sdk" {
        // `OtelConfig::from_env` falls back to "agent-sdk" when
        // OTEL_SERVICE_NAME is unset.  Stamp the host's identity so
        // operators can tell the host binary apart from in-process
        // SDK consumers without having to set the env var.
        otel.service_name = DEFAULT_SERVICE_NAME.to_string();
    }

    if let Some(version) = option_env!("CARGO_PKG_VERSION")
        && otel.service_version.is_none()
    {
        otel.service_version = Some(version.to_string());
    }

    if let Some(instance_id) = &cfg.service_instance_id {
        otel.service_instance_id = Some(instance_id.clone());
    }
    if let Some(env) = &cfg.deployment_environment {
        otel.deployment_environment = Some(env.clone());
    }

    if let Some(endpoint) = &cfg.otlp_endpoint {
        otel.otlp_endpoint = Some(endpoint.clone());
    }
    if !cfg.otlp_headers.is_empty() {
        otel.otlp_headers.clone_from(&cfg.otlp_headers);
    }

    if let Some(sampler_str) = &cfg.sampler {
        let sampler = SamplerKind::from_str(sampler_str)
            .with_context(|| format!("parsing observability.sampler `{sampler_str}`"))?;
        otel.sampler = sampler;
    }
    if let Some(ratio) = cfg.sample_ratio {
        otel.sample_ratio = ratio;
    }
    if !cfg.propagated_baggage_keys.is_empty() {
        otel.propagated_baggage_keys
            .clone_from(&cfg.propagated_baggage_keys);
    }
    otel.capture_payloads = cfg.capture_payloads;

    Ok(otel)
}

/// Register `db.pool.connections.{active,idle}` gauges against the
/// host's Postgres pool when the registry is backed by Postgres.
///
/// The returned `Vec` holds the [`ObservableGauge`] handles that the
/// caller must keep alive — dropping them de-registers the callbacks
/// from the meter provider.  When the registry is not Postgres-backed
/// the function returns an empty `Vec`.
#[must_use]
pub fn install_postgres_pool_gauges(
    stores: &super::stores::StoreRegistry,
) -> Vec<ObservableGauge<u64>> {
    #[cfg(feature = "postgres")]
    {
        let Some(pool) = stores.postgres_pool() else {
            return Vec::new();
        };
        let metrics = HostMetrics::global();
        let (active, idle) =
            metrics.register_postgres_pool_gauges("agent_service_host.postgres", pool.clone());
        vec![active, idle]
    }
    #[cfg(not(feature = "postgres"))]
    {
        // Without the Postgres feature there is no `postgres_pool`
        // method on the registry; suppress the unused-variable
        // warning explicitly.
        let _ = stores;
        Vec::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_grpc_path_extracts_service_and_method() {
        assert_eq!(
            parse_grpc_path("/agent.service.v1.AgentControlService/CreateThread"),
            Some(("agent.service.v1.AgentControlService", "CreateThread")),
        );
    }

    #[test]
    fn parse_grpc_path_rejects_garbage() {
        assert_eq!(parse_grpc_path("/"), None);
        assert_eq!(parse_grpc_path("/foo"), None);
        assert_eq!(parse_grpc_path(""), None);
        assert_eq!(parse_grpc_path("/foo/"), None);
        assert_eq!(parse_grpc_path("//bar"), None);
    }

    #[test]
    fn install_observability_skips_when_disabled() -> anyhow::Result<()> {
        let cfg = super::super::config::ObservabilityConfig::default();
        let guard = install_observability(&cfg)?;
        assert!(guard.is_none(), "disabled config must return Ok(None)");
        Ok(())
    }

    #[test]
    fn build_otel_config_layers_yaml_over_env() -> anyhow::Result<()> {
        let cfg = super::super::config::ObservabilityConfig {
            enabled: true,
            service_name: Some("agent-service-host".into()),
            service_instance_id: Some("host-01".into()),
            deployment_environment: Some("staging".into()),
            otlp_endpoint: Some("http://collector:4317".into()),
            otlp_headers: vec![("authorization".into(), "Bearer secret".into())],
            sampler: Some("parentbased_traceidratio".into()),
            sample_ratio: Some(0.25),
            propagated_baggage_keys: vec!["user.id".into()],
            capture_payloads: true,
        };

        let otel = build_otel_config(&cfg)?;
        assert_eq!(otel.service_name, "agent-service-host");
        assert_eq!(otel.service_instance_id.as_deref(), Some("host-01"));
        assert_eq!(otel.deployment_environment.as_deref(), Some("staging"));
        assert_eq!(otel.endpoint(), Some("http://collector:4317"));
        assert_eq!(otel.otlp_headers.len(), 1);
        assert!((otel.sample_ratio - 0.25).abs() < f64::EPSILON);
        assert_eq!(otel.propagated_baggage_keys, vec!["user.id".to_string()]);
        assert!(otel.capture_payloads);
        Ok(())
    }

    #[test]
    fn build_otel_config_rejects_unknown_sampler() {
        let cfg = super::super::config::ObservabilityConfig {
            enabled: true,
            sampler: Some("definitely-not-a-sampler".into()),
            ..Default::default()
        };
        let Err(err) = build_otel_config(&cfg) else {
            panic!("must reject unknown sampler");
        };
        let chained = format!("{err:#}");
        assert!(
            chained.contains("observability.sampler"),
            "unexpected error: {chained}"
        );
    }
}
