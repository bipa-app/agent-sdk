//! Helpers shared by the conformance tests.
//!
//! The harness covers three concerns:
//!
//! 1. **Provider setup** — install fresh in-memory tracer + meter
//!    pipelines for each test. The `Metrics` singleton is reset so
//!    instruments rebind to the per-test meter provider.
//! 2. **Span lookup** — locate the SDK-emitted spans by name +
//!    conversation id, then pull individual attributes off them.
//! 3. **Metric assertions** — fold the per-test `ResourceMetrics`
//!    snapshot into flat lists of `{labels}` so tests can assert on
//!    "the data point with this label set was recorded".
//!
//! Tests serialize on a dedicated [`TEST_LOCK`] separate from the
//! existing `observability_integration.rs` mutex because the `OTel`
//! provider is process-wide and concurrent tests would race.

#![allow(dead_code)] // Each conformance test pulls a different subset.

use agent_sdk::ThreadId;
use agent_sdk::observability::attrs;
use anyhow::{Context, Result};
use opentelemetry::KeyValue as MetricKv;
use opentelemetry::global;
use opentelemetry::trace::TraceId;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, Sampler, SdkTracerProvider, SpanData};
use tokio::sync::{Mutex, MutexGuard};

/// Tests share the global tracer/meter providers; serialize them.
///
/// Distinct from the `observability_integration.rs` lock so the two
/// files can run in parallel without poisoning each other.
pub static TEST_LOCK: Mutex<()> = Mutex::const_new(());

/// Bundle of providers + their in-memory exporters.
///
/// Returned by [`setup_in_memory_provider`]. Hold the providers for
/// the duration of a test and call `force_flush_all` before reading
/// out spans / metrics.
pub struct InMemoryHarness {
    pub tracer_provider: SdkTracerProvider,
    pub span_exporter: InMemorySpanExporter,
    pub meter_provider: SdkMeterProvider,
    pub metric_exporter: InMemoryMetricExporter,
}

impl InMemoryHarness {
    /// Force-flush both pipelines so every async-emitted span /
    /// metric is visible to the in-memory exporters.
    pub fn force_flush_all(&self) -> Result<()> {
        self.tracer_provider
            .force_flush()
            .context("force_flush tracer provider")?;
        self.meter_provider
            .force_flush()
            .context("force_flush meter provider")?;
        Ok(())
    }

    /// Return every finished span captured so far.
    pub fn spans(&self) -> Result<Vec<SpanData>> {
        self.span_exporter
            .get_finished_spans()
            .context("read finished spans")
    }

    /// Return every metric snapshot captured so far.
    pub fn metrics(&self) -> Result<Vec<ResourceMetrics>> {
        self.metric_exporter
            .get_finished_metrics()
            .context("read finished metrics")
    }
}

pub async fn acquire_test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().await
}

/// Build fresh in-memory tracer + meter providers, install them
/// globally, and reset the SDK `Metrics` singleton so the next
/// `Metrics::global()` call rebinds against this provider.
#[must_use]
pub fn setup_in_memory_provider() -> InMemoryHarness {
    setup_in_memory_provider_with_sampler(Sampler::AlwaysOn)
}

#[must_use]
pub fn setup_in_memory_provider_with_sampler(sampler: Sampler) -> InMemoryHarness {
    let span_exporter = InMemorySpanExporter::default();
    let tracer_provider = SdkTracerProvider::builder()
        .with_sampler(sampler)
        .with_simple_exporter(span_exporter.clone())
        .build();
    global::set_tracer_provider(tracer_provider.clone());

    let metric_exporter = InMemoryMetricExporter::default();
    let meter_provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(metric_exporter.clone()).build())
        .build();
    global::set_meter_provider(meter_provider.clone());

    // Rebind the SDK's `Metrics` singleton to the freshly installed
    // meter provider — otherwise the singleton built in a previous
    // test would point at an exporter we no longer own and every
    // assertion in this test would see zero samples.
    agent_sdk::observability::metrics::Metrics::reset_for_testing();

    InMemoryHarness {
        tracer_provider,
        span_exporter,
        meter_provider,
        metric_exporter,
    }
}

// ── Span helpers ─────────────────────────────────────────────────────

/// Return the `invoke_agent` root span for the supplied thread.
pub fn root_span_for_thread<'a>(
    spans: &'a [SpanData],
    thread_id: &ThreadId,
) -> Result<&'a SpanData> {
    let conversation_id = thread_id.to_string();
    spans
        .iter()
        .find(|span| {
            span.name.as_ref() == "invoke_agent"
                && get_attr(span, attrs::GEN_AI_CONVERSATION_ID).as_deref()
                    == Some(conversation_id.as_str())
        })
        .with_context(|| format!("missing invoke_agent span for thread {conversation_id}"))
}

/// Filter to every span sharing the supplied trace id.
#[must_use]
pub fn spans_in_trace(spans: &[SpanData], trace_id: TraceId) -> Vec<&SpanData> {
    spans
        .iter()
        .filter(|span| span.span_context.trace_id() == trace_id)
        .collect()
}

/// Find a span by exact name inside a borrowed trace slice.
pub fn find_span_in_trace<'a>(spans: &[&'a SpanData], name: &str) -> Result<&'a SpanData> {
    spans
        .iter()
        .copied()
        .find(|span| span.name.as_ref() == name)
        .with_context(|| format!("missing {name} span in trace"))
}

/// Read an attribute by key, returning the `Display`-formatted value
/// (this matches the `OTel` attribute string representation tests use
/// across the SDK).
#[must_use]
pub fn get_attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| format!("{}", kv.value))
}

pub fn assert_span_attribute(span: &SpanData, key: &str, expected: &str) {
    assert_eq!(
        get_attr(span, key).as_deref(),
        Some(expected),
        "expected {key}={expected} on span {:?}",
        span.name,
    );
}

pub fn assert_span_attribute_present(span: &SpanData, key: &str) {
    assert!(
        get_attr(span, key).is_some_and(|v| !v.is_empty()),
        "expected {key} to be present and non-empty on span {:?}",
        span.name,
    );
}

pub fn assert_span_attribute_absent(span: &SpanData, key: &str) {
    assert!(
        get_attr(span, key).is_none(),
        "expected {key} to be absent on span {:?}, got {:?}",
        span.name,
        get_attr(span, key),
    );
}

// ── Metric helpers ───────────────────────────────────────────────────

/// Flatten every `Histogram` data point matching `metric_name` into
/// a list of `{key,value}` label sets across every scope.
#[must_use]
pub fn collect_histogram_attrs(
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
                match metric.data() {
                    AggregatedMetrics::F64(MetricData::Histogram(h)) => {
                        for dp in h.data_points() {
                            out.push(kv_pairs(dp.attributes()));
                        }
                    }
                    AggregatedMetrics::U64(MetricData::Histogram(h)) => {
                        for dp in h.data_points() {
                            out.push(kv_pairs(dp.attributes()));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

/// Counter equivalent of [`collect_histogram_attrs`]. Counters land
/// in the `Sum` variant; we fold every label vector so tests assert
/// on the label combinations the recorder produced.
#[must_use]
pub fn collect_counter_attrs(
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
                match metric.data() {
                    AggregatedMetrics::U64(MetricData::Sum(sum)) => {
                        for dp in sum.data_points() {
                            out.push(kv_pairs(dp.attributes()));
                        }
                    }
                    AggregatedMetrics::F64(MetricData::Sum(sum)) => {
                        for dp in sum.data_points() {
                            out.push(kv_pairs(dp.attributes()));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

/// Assert at least one histogram data point matches every `(key,value)` in
/// `expected`.
pub fn assert_metric_histogram_sample(
    snapshots: &[ResourceMetrics],
    metric_name: &str,
    expected: &[(&str, &str)],
) {
    let points = collect_histogram_attrs(snapshots, metric_name);
    assert!(
        points.iter().any(|p| matches_all(p, expected)),
        "missing histogram sample for {metric_name} with {expected:?}; got {points:?}",
    );
}

/// Assert at least one counter data point matches every `(key,value)` in
/// `expected`.
pub fn assert_metric_counter_sample(
    snapshots: &[ResourceMetrics],
    metric_name: &str,
    expected: &[(&str, &str)],
) {
    let points = collect_counter_attrs(snapshots, metric_name);
    assert!(
        points.iter().any(|p| matches_all(p, expected)),
        "missing counter sample for {metric_name} with {expected:?}; got {points:?}",
    );
}

fn kv_pairs<'a>(iter: impl Iterator<Item = &'a MetricKv>) -> Vec<(String, String)> {
    iter.map(|kv| (kv.key.as_str().to_string(), format!("{}", kv.value)))
        .collect()
}

#[must_use]
pub fn has_label(set: &[(String, String)], key: &str, value: &str) -> bool {
    set.iter()
        .any(|(k, v)| k.as_str() == key && v.as_str() == value)
}

#[must_use]
pub fn matches_all(set: &[(String, String)], expected: &[(&str, &str)]) -> bool {
    expected.iter().all(|(k, v)| has_label(set, k, v))
}

// ── Payload-capture gate guard ───────────────────────────────────────

/// RAII guard that flips the SDK's process-wide payload-capture gate
/// for the lifetime of the binding and restores the previous value on
/// drop. The C2 gate lives in a `static AtomicBool`; without this
/// guard a panicking test would leak the flipped value to every
/// subsequent test on the same binary.
pub struct CaptureGateGuard {
    previous: bool,
}

impl CaptureGateGuard {
    #[must_use]
    pub fn set(enabled: bool) -> Self {
        let previous = agent_sdk::observability::is_payload_capture_enabled();
        agent_sdk::observability::set_payload_capture_enabled(enabled);
        Self { previous }
    }
}

impl Drop for CaptureGateGuard {
    fn drop(&mut self) {
        agent_sdk::observability::set_payload_capture_enabled(self.previous);
    }
}

// ── Run-helper ───────────────────────────────────────────────────────

/// Wait for the agent's final-state future, then yield long enough
/// for any post-state span emissions to drain through the exporter.
pub async fn wait_for_run(
    final_state: impl std::future::Future<Output = Result<agent_sdk::AgentRunState>>,
) -> Result<()> {
    let _ = final_state
        .await
        .context("agent run did not report a state")?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    Ok(())
}
