//! Bootstrap helpers for installing OpenTelemetry tracing + metrics into a
//! process that consumes the [Agent SDK](https://docs.rs/agent-sdk).
//!
//! The whole crate is one decision: should we install an OTLP gRPC pipeline
//! or run silently? Callers express the answer through [`OtelConfig`] and
//! call [`install_global_provider`] once at startup. The returned
//! [`OtelGuard`] flushes pending exports on `Drop` (or via the explicit
//! [`OtelGuard::shutdown`] entry point), so failure to drop becomes a
//! shutdown leak, not a span/metric leak.
//!
//! ## Quick start
//!
//! ```no_run
//! use agent_sdk_otel::{OtelConfig, install_global_provider};
//!
//! fn main() -> anyhow::Result<()> {
//!     let cfg = OtelConfig::from_env()?;
//!     let _guard = install_global_provider(&cfg)?;
//!     // ... run the agent here ...
//!     Ok(())
//! }
//! ```
//!
//! ## Lean dependency footprint
//!
//! `opentelemetry-otlp` pulls `tonic`, `prost`, `tokio-rustls`, etc. The
//! main `agent-sdk` crate stays free of those by keeping the exporter
//! wiring here. Consumers that don't need OTLP can simply not depend on
//! `agent-sdk-otel`.

mod config;
mod propagator;
mod resource;
mod sampler;

use anyhow::{Context, Result};
use opentelemetry::global;
use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig, WithTonicConfig};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;

pub use config::{OtelConfig, OtelConfigBuilder};
pub use propagator::AllowListBaggagePropagator;
pub use sampler::SamplerKind;

/// Drop guard returned by [`install_global_provider`].
///
/// Holding the guard keeps the global tracer/meter providers alive. Drop
/// or call [`OtelGuard::shutdown`] to flush pending exports and tear down
/// the `OTel` pipeline.
#[must_use = "drop the guard at program shutdown to flush exports"]
pub struct OtelGuard {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl OtelGuard {
    /// Flush all pending spans + metrics and shut the providers down.
    ///
    /// Equivalent to dropping the guard, but propagates the first
    /// shutdown error rather than swallowing it. Subsequent shutdowns
    /// are no-ops.
    ///
    /// # Errors
    /// Returns the first error encountered while shutting down either
    /// provider. The other provider is still shut down before the
    /// function returns.
    pub fn shutdown(mut self) -> Result<()> {
        let trace_err = self.tracer_provider.take().and_then(|p| p.shutdown().err());
        let meter_err = self.meter_provider.take().and_then(|p| p.shutdown().err());
        match (trace_err, meter_err) {
            (None, None) => Ok(()),
            (Some(err), None) => Err(err).context("tracer provider shutdown failed"),
            (None, Some(err)) => Err(err).context("meter provider shutdown failed"),
            (Some(trace_err), Some(meter_err)) => Err(trace_err).context(format!(
                "tracer provider shutdown failed (meter shutdown also failed: {meter_err})"
            )),
        }
    }
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.tracer_provider.take()
            && let Err(err) = provider.shutdown()
        {
            log::warn!(
                target: "agent_sdk_otel",
                "tracer provider shutdown failed during drop: {err}"
            );
        }
        if let Some(provider) = self.meter_provider.take()
            && let Err(err) = provider.shutdown()
        {
            log::warn!(
                target: "agent_sdk_otel",
                "meter provider shutdown failed during drop: {err}"
            );
        }
    }
}

/// Install the global tracer + meter providers and propagator.
///
/// Reads from the supplied [`OtelConfig`] only — no environment-variable
/// reads happen here, so callers must funnel env config through
/// [`OtelConfig::from_env`] (or build it manually). When
/// [`OtelConfig::exporter_enabled`] is `false`, an in-memory no-op pipeline
/// is installed: span/metric calls succeed, nothing is exported.
///
/// # Errors
/// Returns an error if the OTLP exporter cannot be built — the most
/// common cause is a malformed endpoint URL.
pub fn install_global_provider(cfg: &OtelConfig) -> Result<OtelGuard> {
    // Phase 9 · C2: flip the SDK's process-wide payload-capture
    // gate to match the operator's choice.  The gate defaults to
    // closed; flipping it open still requires every store to
    // override `ObservabilityStore::acknowledge_pii_redaction()` to
    // return true before any payload reaches a span inline.
    agent_sdk::observability::set_payload_capture_enabled(cfg.capture_payloads);
    if cfg.capture_payloads {
        log::warn!(
            target: "agent_sdk_otel",
            "agent_sdk observability: payload capture is ENABLED. Stores must override \
             ObservabilityStore::acknowledge_pii_redaction() to return true *and* \
             install a non-noop PayloadRedactor for `gen_ai.input.messages` / \
             `gen_ai.output.messages` to land on spans."
        );
    }

    let resource = resource::build(cfg);
    let sampler = sampler::resolve(cfg.sampler, cfg.sample_ratio)
        .context("failed to resolve OTel sampler")?;

    let (tracer_provider, meter_provider) = if cfg.exporter_enabled() {
        build_exporting_providers(cfg, resource, sampler)?
    } else {
        log::info!(
            target: "agent_sdk_otel",
            "OTLP exporter disabled (OTEL_EXPORTER_OTLP_ENDPOINT empty/unset); installing no-op providers"
        );
        let trace_provider = SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_sampler(sampler)
            .build();
        let meter_provider = SdkMeterProvider::builder().with_resource(resource).build();
        (trace_provider, meter_provider)
    };

    global::set_tracer_provider(tracer_provider.clone());
    global::set_meter_provider(meter_provider.clone());

    // Phase 9 · C3: wrap the upstream `BaggagePropagator` with an
    // exact-match allow-list so non-allow-listed baggage entries
    // never leave the process. An empty
    // `cfg.propagated_baggage_keys` falls back to the SDK baseline
    // (the five keys from `agent_sdk::observability::baggage`).
    let baggage_keys = if cfg.propagated_baggage_keys.is_empty() {
        AllowListBaggagePropagator::baseline_allow_list()
    } else {
        cfg.propagated_baggage_keys.clone()
    };
    let baggage_propagator = AllowListBaggagePropagator::new(baggage_keys);
    let propagator = opentelemetry::propagation::TextMapCompositePropagator::new(vec![
        Box::new(TraceContextPropagator::new()),
        Box::new(baggage_propagator),
    ]);
    global::set_text_map_propagator(propagator);

    Ok(OtelGuard {
        tracer_provider: Some(tracer_provider),
        meter_provider: Some(meter_provider),
    })
}

fn build_exporting_providers(
    cfg: &OtelConfig,
    resource: opentelemetry_sdk::Resource,
    sampler: opentelemetry_sdk::trace::Sampler,
) -> Result<(SdkTracerProvider, SdkMeterProvider)> {
    let endpoint = cfg.endpoint().context("exporter endpoint missing")?;
    let metadata =
        build_metadata(&cfg.otlp_headers).context("failed to assemble OTLP gRPC metadata")?;

    let span_builder = SpanExporter::builder().with_tonic().with_endpoint(endpoint);
    let span_builder = span_builder.with_metadata(metadata.clone());
    let span_exporter = span_builder
        .build()
        .context("failed to build OTLP span exporter")?;

    let metric_builder = MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint);
    let metric_builder = metric_builder.with_metadata(metadata);
    let metric_exporter = metric_builder
        .build()
        .context("failed to build OTLP metric exporter")?;

    let tracer_provider = SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_sampler(sampler)
        .with_batch_exporter(span_exporter)
        .build();

    let meter_provider = SdkMeterProvider::builder()
        .with_resource(resource)
        .with_periodic_exporter(metric_exporter)
        .build();

    Ok((tracer_provider, meter_provider))
}

fn build_metadata(
    headers: &[(String, String)],
) -> Result<opentelemetry_otlp::tonic_types::metadata::MetadataMap> {
    use opentelemetry_otlp::tonic_types::metadata::MetadataMap;
    use tonic::metadata::{MetadataKey, MetadataValue};

    let mut map = MetadataMap::with_capacity(headers.len());
    for (key, value) in headers {
        let metadata_key: MetadataKey<_> = key
            .parse()
            .with_context(|| format!("invalid OTLP header key `{key}`"))?;
        let metadata_value: MetadataValue<_> = value
            .parse()
            .with_context(|| format!("invalid OTLP header value for `{key}`"))?;
        map.insert(metadata_key, metadata_value);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_metadata_rejects_bad_keys() {
        let Err(err) = build_metadata(&[("bad key".to_string(), "value".to_string())]) else {
            panic!("expected invalid header key");
        };
        assert!(format!("{err}").contains("invalid OTLP header key"));
    }

    #[test]
    fn build_metadata_accepts_well_formed_headers() -> Result<()> {
        let map = build_metadata(&[
            ("authorization".to_string(), "Bearer secret".to_string()),
            ("x-tenant".to_string(), "luiz".to_string()),
        ])?;
        assert_eq!(map.len(), 2);
        Ok(())
    }
}
