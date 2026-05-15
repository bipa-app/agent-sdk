//! Integration coverage for Phase 9 · E1: the host's adoption of
//! [`agent_sdk_otel::install_global_provider`] plus the
//! `db.pool.connections.{active,idle}` gauge wiring through
//! [`ServiceHost::new`].
//!
//! Compiled only with `--features otel`; without the feature the
//! host's observability install path is gated out entirely.

#![cfg(feature = "otel")]

use std::sync::Arc;
use std::time::Duration;

use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
use agent_service_host::config::{ObservabilityConfig, ServiceConfig};
use agent_service_host::host::ServiceHost;
use agent_service_host::observability::{HostMetrics, install_observability};
use agent_service_host::runtime::{
    AllowAllConfirmationPolicy, ExecutionRuntime, NoopToolExecutor, StaticProviderResolver,
};
use anyhow::{Context, Result};
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use tokio::sync::{Mutex, MutexGuard};

// ─────────────────────────────────────────────────────────────────────
// Test plumbing
// ─────────────────────────────────────────────────────────────────────

/// Tests in this file mutate the process-wide `MeterProvider` /
/// `HostMetrics` singleton; serialise them so concurrent test
/// threads do not stomp on each other.  Async-aware mutex so the
/// guard can be held across `await` points.
static TEST_LOCK: Mutex<()> = Mutex::const_new(());

async fn acquire_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().await
}

fn install_in_memory_meter() -> (SdkMeterProvider, InMemoryMetricExporter) {
    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    HostMetrics::reset_for_testing();
    (provider, exporter)
}

fn flush(provider: &SdkMeterProvider) -> Result<()> {
    provider.force_flush().context("flush meter provider")?;
    Ok(())
}

fn collected(exporter: &InMemoryMetricExporter) -> Result<Vec<ResourceMetrics>> {
    exporter
        .get_finished_metrics()
        .context("read collected metrics")
}

fn collect_u64_gauge(
    snapshots: &[ResourceMetrics],
    metric_name: &str,
) -> Vec<(Vec<(String, String)>, u64)> {
    let mut out = Vec::new();
    for resource in snapshots {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != metric_name {
                    continue;
                }
                if let AggregatedMetrics::U64(MetricData::Gauge(g)) = metric.data() {
                    for dp in g.data_points() {
                        out.push((kv_pairs(dp.attributes()), dp.value()));
                    }
                }
            }
        }
    }
    out
}

fn kv_pairs<'a>(iter: impl Iterator<Item = &'a KeyValue>) -> Vec<(String, String)> {
    iter.map(|kv| (kv.key.as_str().to_string(), format!("{}", kv.value)))
        .collect()
}

fn sample_definition() -> AgentDefinition {
    AgentDefinition {
        provider: "mock".into(),
        model: "mock-model".into(),
        system_prompt: "test".into(),
        max_tokens: 4096,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        tools_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

fn sample_runtime() -> Arc<ExecutionRuntime> {
    Arc::new(ExecutionRuntime::new(
        Arc::new(StaticProviderResolver::new()),
        Arc::new(NoopToolExecutor),
        Arc::new(AllowAllConfirmationPolicy),
    ))
}

// ─────────────────────────────────────────────────────────────────────
// 1. install_observability: skip-when-disabled + endpoint-empty smoke
// ─────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn install_observability_returns_none_when_disabled() -> Result<()> {
    let _guard = acquire_lock().await;

    let cfg = ObservabilityConfig::default();
    let outcome = install_observability(&cfg)?;
    assert!(
        outcome.is_none(),
        "default config (enabled=false) must return Ok(None)",
    );
    Ok(())
}

/// `enabled = true` with an explicitly empty endpoint installs a
/// no-op pipeline that still hands back a guard. This is the
/// "smoke-test the install path with no collector running"
/// scenario the host docs call out.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn install_observability_with_empty_endpoint_returns_guard() -> Result<()> {
    let _guard = acquire_lock().await;

    let cfg = ObservabilityConfig {
        enabled: true,
        otlp_endpoint: Some(String::new()),
        ..Default::default()
    };
    let Some(otel_guard) = install_observability(&cfg)? else {
        anyhow::bail!("expected Some(OtelGuard) when enabled=true");
    };
    otel_guard.shutdown()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 2. ServiceHost::new wires Postgres pool gauges (live Postgres only)
// ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "postgres")]
#[tokio::test]
#[ignore = "requires TEST_DATABASE_URL"]
async fn service_host_registers_postgres_pool_gauges() -> Result<()> {
    let _guard = acquire_lock().await;

    let Ok(database_url) =
        std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
    else {
        // Mirror the existing host_metrics.rs gating: skip cleanly
        // on hosts without the test database wired up.
        return Ok(());
    };

    let (provider, exporter) = install_in_memory_meter();

    let config = ServiceConfig {
        storage: agent_service_host::config::StorageConfig {
            backend: agent_service_host::config::StorageBackend::Postgres,
            postgres: agent_service_host::config::PostgresStorageConfig {
                database_url: Some(database_url.clone()),
                schema: None,
                max_connections: 2,
            },
        },
        ..Default::default()
    };
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(sample_definition()));
    let host = ServiceHost::new(config, registry, sample_runtime())
        .context("construct ServiceHost with Postgres backend")?;

    // Force one connection in the pool so the active gauge sees a
    // non-zero observation.
    if let Some(pool) = host.stores().postgres_pool() {
        let _conn = pool.acquire().await.context("acquire test connection")?;
    }
    tokio::time::sleep(Duration::from_millis(20)).await;

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let active_points = collect_u64_gauge(&snapshots, "db.pool.connections.active");
    let idle_points = collect_u64_gauge(&snapshots, "db.pool.connections.idle");

    assert!(
        !active_points.is_empty(),
        "ServiceHost::new must register db.pool.connections.active",
    );
    assert!(
        !idle_points.is_empty(),
        "ServiceHost::new must register db.pool.connections.idle",
    );

    Ok(())
}
