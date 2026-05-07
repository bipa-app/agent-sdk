//! Integration tests for `crates/agent-service-host/src/observability.rs`.
//!
//! Covers the `gRPC` `tower::Layer`, the `OTel` record helpers, and
//! the Postgres pool gauge wiring. `AMQP`-broker tests live
//! alongside the existing AMQP integration suite and are
//! `#[ignore]`-gated like the rest.
//!
//! The file compiles only with `--features otel`; without the
//! feature the host's observability module isn't even on disk.

#![cfg(feature = "otel")]

use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::{Context as _, Result};
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use tokio::sync::{Mutex, MutexGuard};
use tower::{Layer, Service};

use agent_service_host::observability::grpc_layer::MetricsLayer;
use agent_service_host::observability::{HostMetrics, attrs, parse_grpc_path};

// ─────────────────────────────────────────────────────────────────────
// Harness
// ─────────────────────────────────────────────────────────────────────

static TEST_LOCK: Mutex<()> = Mutex::const_new(());

async fn acquire_test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().await
}

fn setup_meter() -> (SdkMeterProvider, InMemoryMetricExporter) {
    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    HostMetrics::reset_for_testing();
    (provider, exporter)
}

fn collected(exporter: &InMemoryMetricExporter) -> Result<Vec<ResourceMetrics>> {
    exporter
        .get_finished_metrics()
        .context("read collected metrics")
}

fn flush(provider: &SdkMeterProvider) -> Result<()> {
    provider.force_flush().context("flush meter provider")?;
    Ok(())
}

fn kv_pairs<'a>(iter: impl Iterator<Item = &'a KeyValue>) -> Vec<(String, String)> {
    iter.map(|kv| (kv.key.as_str().to_string(), format!("{}", kv.value)))
        .collect()
}

fn collect_histogram_attrs(
    snapshots: &[ResourceMetrics],
    metric_name: &str,
) -> Vec<Vec<(String, String)>> {
    let mut out = Vec::new();
    for resource in snapshots {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != metric_name {
                    continue;
                }
                if let AggregatedMetrics::F64(MetricData::Histogram(h)) = metric.data() {
                    for dp in h.data_points() {
                        out.push(kv_pairs(dp.attributes()));
                    }
                }
            }
        }
    }
    out
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

fn has_label(set: &[(String, String)], key: &str, value: &str) -> bool {
    set.iter().any(|(k, v)| k == key && v == value)
}

fn matches_all(set: &[(String, String)], expected: &[(&str, &str)]) -> bool {
    expected.iter().all(|(k, v)| has_label(set, k, v))
}

// ─────────────────────────────────────────────────────────────────────
// Mock tonic-shaped tower service
// ─────────────────────────────────────────────────────────────────────

/// Minimal `tower::Service` that swallows the request and returns
/// the configured response. Mirrors the `http::Request<B>` →
/// `http::Response<Body>` shape of a tonic server.
#[derive(Clone)]
struct MockService {
    response: http::Response<()>,
}

impl MockService {
    fn ok() -> Self {
        // Trailers-only OK responses (the normal happy path) put
        // grpc-status in trailers, not headers; the layer should
        // record `0` for both since absent header means OK.
        Self {
            response: http::Response::new(()),
        }
    }

    fn header_status(code: i32) -> Self {
        let mut response = http::Response::new(());
        response
            .headers_mut()
            .insert("grpc-status", code.to_string().parse().unwrap());
        Self { response }
    }
}

impl<B> Service<http::Request<B>> for MockService
where
    B: Send + 'static,
{
    type Response = http::Response<()>;
    type Error = Infallible;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: http::Request<B>) -> Self::Future {
        let response = self.response.clone();
        Box::pin(async move { Ok(response) })
    }
}

fn build_request(path: &str) -> http::Request<()> {
    http::Request::builder().uri(path).body(()).unwrap()
}

// ─────────────────────────────────────────────────────────────────────
// 1. parse_grpc_path
// ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_grpc_path_extracts_service_and_method() {
    assert_eq!(
        parse_grpc_path("/agent.service.v1.AgentControlService/CreateThread"),
        Some(("agent.service.v1.AgentControlService", "CreateThread")),
    );
}

#[test]
fn parse_grpc_path_rejects_garbage_paths() {
    for bad in ["", "/", "/foo", "//bar", "/foo/", "no-leading-slash"] {
        assert_eq!(parse_grpc_path(bad), None, "expected None for {bad:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// 2. MetricsLayer records rpc.server.duration with full attribute set
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn metrics_layer_records_rpc_server_duration_with_ok_status() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    // Force-init the singleton against this provider before the
    // layer touches it.
    let _ = HostMetrics::global();

    let layer = MetricsLayer::new();
    let mut svc = layer.layer(MockService::ok());
    let response = svc
        .call(build_request(
            "/agent.service.v1.AgentControlService/CreateThread",
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), http::StatusCode::OK);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "rpc.server.duration");
    assert!(!points.is_empty(), "no rpc.server.duration sample recorded");
    for p in &points {
        assert!(matches_all(
            p,
            &[
                (attrs::RPC_SYSTEM, "grpc"),
                (attrs::RPC_SERVICE, "agent.service.v1.AgentControlService"),
                (attrs::RPC_METHOD, "CreateThread"),
                (attrs::RPC_GRPC_STATUS_CODE, "0"),
            ],
        ));
    }
    Ok(())
}

#[tokio::test]
async fn metrics_layer_propagates_grpc_status_from_response_headers() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let _ = HostMetrics::global();

    let layer = MetricsLayer::new();
    // 3 = INVALID_ARGUMENT
    let mut svc = layer.layer(MockService::header_status(3));
    let _ = svc
        .call(build_request(
            "/agent.service.v1.AgentControlService/SubmitWork",
        ))
        .await
        .unwrap();

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "rpc.server.duration");
    assert!(
        points.iter().any(|p| matches_all(
            p,
            &[
                (attrs::RPC_METHOD, "SubmitWork"),
                (attrs::RPC_GRPC_STATUS_CODE, "3"),
            ],
        )),
        "expected sample with grpc status 3, got {points:?}",
    );
    Ok(())
}

#[tokio::test]
async fn metrics_layer_skips_paths_without_service_and_method() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let _ = HostMetrics::global();

    let layer = MetricsLayer::new();
    let mut svc = layer.layer(MockService::ok());
    // A bare `/` cannot be parsed into service/method — the layer
    // must skip rather than recording an empty-attribute sample.
    let _ = svc.call(build_request("/")).await.unwrap();

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "rpc.server.duration");
    assert!(
        points.is_empty(),
        "no points expected for a path without service/method, got {points:?}",
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 3. AMQP publish/consume helpers
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn amqp_publish_records_per_outcome() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = HostMetrics::global();

    metrics.record_amqp_publish("agent_sdk.outbox", 0.012, true);
    metrics.record_amqp_publish("agent_sdk.outbox", 0.250, false);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "agent_service_host.amqp.publish.duration");
    assert!(points.iter().any(|p| matches_all(
        p,
        &[
            (attrs::AMQP_DESTINATION, "agent_sdk.outbox"),
            (attrs::AMQP_OUTCOME, attrs::AMQP_OUTCOME_OK),
        ],
    )));
    assert!(points.iter().any(|p| matches_all(
        p,
        &[
            (attrs::AMQP_DESTINATION, "agent_sdk.outbox"),
            (attrs::AMQP_OUTCOME, attrs::AMQP_OUTCOME_ERROR),
        ],
    )));
    Ok(())
}

#[tokio::test]
async fn amqp_consume_records_ack_and_nack() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = HostMetrics::global();

    metrics.record_amqp_consume("agent_sdk.task_wakeup", 0.005, true);
    metrics.record_amqp_consume("agent_sdk.task_wakeup", 0.030, false);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "agent_service_host.amqp.consume.duration");
    assert!(
        points
            .iter()
            .any(|p| matches_all(p, &[(attrs::AMQP_OUTCOME, attrs::AMQP_OUTCOME_ACK)],))
    );
    assert!(
        points
            .iter()
            .any(|p| matches_all(p, &[(attrs::AMQP_OUTCOME, attrs::AMQP_OUTCOME_NACK)],))
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 4. AMQP queue-depth gauge
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn amqp_queue_depth_gauge_publishes_callback_value() -> Result<()> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = HostMetrics::global();

    let depth = Arc::new(AtomicU64::new(11));
    let depth_for_callback = Arc::clone(&depth);
    let _gauge = metrics.register_amqp_queue_depth_gauge("agent_sdk.task_wakeup", move || {
        depth_for_callback.load(Ordering::Relaxed)
    });

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_u64_gauge(&snapshots, "agent_service_host.amqp.queue.depth");
    assert!(points.iter().any(|(_, v)| *v == 11));

    depth.store(33, Ordering::Relaxed);
    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_u64_gauge(&snapshots, "agent_service_host.amqp.queue.depth");
    assert!(points.iter().any(|(_, v)| *v == 33));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 5. DB connection-create helper
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn db_connection_create_helper_records_pool_name_and_system() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = HostMetrics::global();

    metrics.record_db_client_connection_create_time("test_pool", 0.003);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "db.client.connections.create_time");
    assert!(points.iter().any(|p| matches_all(
        p,
        &[
            (attrs::DB_SYSTEM, attrs::DB_SYSTEM_POSTGRESQL),
            (attrs::DB_POOL_NAME, "test_pool"),
        ],
    )));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 6. Postgres pool gauges (live Postgres only — gated like the
// other agent-service-host postgres tests)
// ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "postgres")]
#[tokio::test]
#[ignore = "requires TEST_DATABASE_URL"]
async fn postgres_pool_gauges_publish_active_and_idle_counts() -> Result<()> {
    let _guard = acquire_test_lock().await;

    let Ok(database_url) =
        std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
    else {
        return Ok(());
    };

    let (provider, exporter) = setup_meter();
    let metrics = HostMetrics::global();

    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(2)
        .connect(&database_url)
        .await
        .context("connect test pool")?;

    let _gauges = metrics.register_postgres_pool_gauges("test_pool", pool.clone());

    // Force at least one connection in the pool.
    let _conn = pool.acquire().await.context("acquire one connection")?;
    tokio::time::sleep(Duration::from_millis(20)).await;

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let active_points = collect_u64_gauge(&snapshots, "db.pool.connections.active");
    let idle_points = collect_u64_gauge(&snapshots, "db.pool.connections.idle");
    assert!(!active_points.is_empty());
    assert!(!idle_points.is_empty());
    Ok(())
}
