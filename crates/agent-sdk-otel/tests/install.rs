//! Smoke tests for the bootstrap helper.
//!
//! Each test installs a fresh global tracer/meter provider, so we
//! serialise them on a process-wide mutex.

use agent_sdk_otel::{OtelConfig, SamplerKind, install_global_provider};
use anyhow::{Context, Result};
use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::trace::{Span, SpanKind, Status, Tracer, TracerProvider as _};
use std::sync::Mutex;

/// Globally serialise tests that mutate the `OTel` global providers. The
/// underlying `set_tracer_provider` / `set_meter_provider` calls overwrite
/// process-wide state, so running concurrently produces flaky behaviour.
static GLOBAL: Mutex<()> = Mutex::new(());

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn install_with_disabled_exporter_emits_and_shuts_down_cleanly() -> Result<()> {
    let _g = GLOBAL.lock().ok().context("test mutex poisoned")?;

    let cfg = OtelConfig::builder("agent-sdk-otel-smoke")
        .service_version(env!("CARGO_PKG_VERSION"))
        .deployment_environment("test")
        .otlp_endpoint(None)
        .sampler(SamplerKind::AlwaysOn)
        .build();

    let guard = install_global_provider(&cfg)?;

    let tracer = global::tracer_provider().tracer("install-test");
    let mut span = tracer
        .span_builder("smoke")
        .with_kind(SpanKind::Internal)
        .start(&tracer);
    span.set_attribute(KeyValue::new("smoke.kind", "disabled-endpoint"));
    span.set_status(Status::Ok);
    span.end();

    let meter = global::meter_provider().meter("install-test");
    let counter = meter.u64_counter("smoke.calls").build();
    counter.add(1, &[KeyValue::new("kind", "disabled-endpoint")]);

    guard.shutdown()?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn install_with_explicitly_empty_endpoint_disables_exporter() -> Result<()> {
    let _g = GLOBAL.lock().ok().context("test mutex poisoned")?;

    // `OTEL_EXPORTER_OTLP_ENDPOINT=` (empty string) must be treated as
    // "disabled" without panicking — this is one of the explicit
    // acceptance criteria.
    let cfg = OtelConfig::builder("agent-sdk-otel-smoke")
        .otlp_endpoint(Some(String::new()))
        .sampler(SamplerKind::AlwaysOff)
        .build();

    assert!(!cfg.exporter_enabled());
    let guard = install_global_provider(&cfg)?;
    guard.shutdown()?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn install_with_grpc_endpoint_succeeds() -> Result<()> {
    let _g = GLOBAL.lock().ok().context("test mutex poisoned")?;

    // The endpoint points at a closed loopback port; with AlwaysOff
    // sampling we never produce spans, and metric pushes go nowhere
    // before the configured timeout — but the install + shutdown path
    // must still succeed without panicking.
    let cfg = OtelConfig::builder("agent-sdk-otel-smoke")
        .service_version(env!("CARGO_PKG_VERSION"))
        .deployment_environment("test")
        .otlp_endpoint(Some("http://127.0.0.1:1".to_string()))
        .sampler(SamplerKind::AlwaysOff)
        .build();

    let guard = install_global_provider(&cfg)?;

    let tracer = global::tracer_provider().tracer("install-test");
    let mut span = tracer.start("never-sampled");
    span.set_attribute(KeyValue::new("kind", "always-off"));
    span.end();

    guard.shutdown()?;
    Ok(())
}
