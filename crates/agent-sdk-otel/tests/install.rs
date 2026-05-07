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

/// Phase 9 · C2: `install_global_provider` must propagate the
/// operator-level `capture_payloads` flag onto the SDK's
/// process-wide gate so `agent_sdk::observability` knows whether
/// `Inline` payload decisions can land on spans. The gate is OFF
/// by default and only flips when the config opts in.
struct GateGuard(bool);

impl Drop for GateGuard {
    fn drop(&mut self) {
        agent_sdk::observability::set_payload_capture_enabled(self.0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn install_propagates_capture_payloads_flag_onto_sdk_gate() -> Result<()> {
    let _g = GLOBAL.lock().ok().context("test mutex poisoned")?;

    // Restore whatever the gate was set to before the test so
    // adjacent tests are not poisoned. The smoke tests above do
    // not touch the gate, so the natural starting state is `false`,
    // but we still capture and restore explicitly for safety.
    let _restore = GateGuard(agent_sdk::observability::is_payload_capture_enabled());

    // capture_payloads = false (default): the gate must end up closed.
    let cfg_off = OtelConfig::builder("agent-sdk-otel-smoke")
        .otlp_endpoint(None)
        .sampler(SamplerKind::AlwaysOff)
        .build();
    let guard_off = install_global_provider(&cfg_off)?;
    assert!(
        !agent_sdk::observability::is_payload_capture_enabled(),
        "default capture_payloads=false must leave the SDK gate closed",
    );
    guard_off.shutdown()?;

    // capture_payloads = true: the gate must end up open.
    let cfg_on = OtelConfig::builder("agent-sdk-otel-smoke")
        .otlp_endpoint(None)
        .sampler(SamplerKind::AlwaysOff)
        .capture_payloads(true)
        .build();
    let guard_on = install_global_provider(&cfg_on)?;
    assert!(
        agent_sdk::observability::is_payload_capture_enabled(),
        "capture_payloads=true must flip the SDK gate open",
    );
    guard_on.shutdown()?;
    Ok(())
}
