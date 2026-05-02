//! Public configuration surface for the bootstrap helper.
//!
//! [`OtelConfig`] is the value `install_global_provider` consumes; the
//! [`OtelConfigBuilder`] gives callers a fluent way to assemble one in code
//! and `OtelConfig::from_env()` populates it from the standard `OTel`
//! environment variables.

// Project rule (`CLAUDE.md` / Pending Learnings): never use
// `unwrap_or*` / `map_or*` to silently substitute a fallback value. Use
// explicit `match` / `if let` instead and let the fallback site announce
// itself (a log line, a labelled default constant, etc.). The
// `option_if_let_else` and `single_match_else` lints actively push us
// *toward* `map_or_else`, so we silence them module-wide and rely on
// the project rule to keep fallbacks loud.
#![allow(clippy::option_if_let_else, clippy::single_match_else)]

use std::env;
use std::fmt;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use opentelemetry::KeyValue;

use crate::sampler::SamplerKind;

/// Default deployment environment when nothing is configured.
const DEFAULT_DEPLOYMENT_ENVIRONMENT: &str = "unknown_deployment";

/// Default service name used by [`OtelConfig::from_env`] when
/// `OTEL_SERVICE_NAME` is unset. The bootstrap helper logs a warning so the
/// caller knows the resource is unattributed.
const DEFAULT_SERVICE_NAME_FALLBACK: &str = "agent-sdk";

/// Configuration for the `OTel` bootstrap helper.
///
/// Construct via [`OtelConfig::builder`] (programmatic) or
/// [`OtelConfig::from_env`] (12-factor style).
///
/// `Debug` is implemented manually so that `otlp_headers` values are masked
/// — header values frequently carry auth tokens that should never end up in
/// logs.
#[derive(Clone)]
pub struct OtelConfig {
    /// Logical service identity. Becomes `service.name` on the resource.
    pub service_name: String,
    /// Optional explicit `service.version`. Caller-provided to avoid the
    /// trap of using the bootstrap crate's `CARGO_PKG_VERSION` as a default
    /// when the binary's version is what's actually meaningful.
    pub service_version: Option<String>,
    /// Optional explicit `service.instance.id`. Defaults to a fresh UUID v7
    /// when unset, ensuring per-process uniqueness.
    pub service_instance_id: Option<String>,
    /// Optional `deployment.environment` override. Defaults to
    /// `unknown_deployment` when unset.
    pub deployment_environment: Option<String>,
    /// OTLP gRPC endpoint. `None` (or empty string) disables the exporter:
    /// the bootstrap helper still installs an [`opentelemetry_sdk`] provider
    /// so the rest of the SDK can run, but no spans/metrics leave the
    /// process.
    pub otlp_endpoint: Option<String>,
    /// Headers attached to every OTLP request. Carries auth (e.g.
    /// `authorization=Bearer …`) so the manual `Debug` impl masks values.
    pub otlp_headers: Vec<(String, String)>,
    /// Sampler strategy.
    pub sampler: SamplerKind,
    /// Ratio for ratio-based samplers (`TraceIdRatio` /
    /// `ParentBasedTraceIdRatio`). Ignored otherwise.
    pub sample_ratio: f64,
    /// Baggage keys the propagator is allowed to forward across
    /// process boundaries. The bootstrap helper currently installs the
    /// upstream `BaggagePropagator` unfiltered; Track A3/C3 will wrap it
    /// with an allow-list using these keys.
    pub propagated_baggage_keys: Vec<String>,
    /// Whether `gen_ai.input.messages` / `gen_ai.output.messages` /
    /// `gen_ai.system_instructions` payloads may be captured on spans.
    /// The bootstrap helper itself does not consult this — it simply
    /// surfaces the flag so the rest of the SDK can read a single source
    /// of truth. See Track C2.
    pub capture_payloads: bool,
    /// Extra `Resource` attributes appended after the SDK defaults.
    /// Useful for `OTEL_RESOURCE_ATTRIBUTES` parity.
    pub additional_resource_attrs: Vec<KeyValue>,
}

impl OtelConfig {
    /// Start a builder with sensible defaults (sample ratio 1.0,
    /// parent-based traceid sampler, no exporter).
    #[must_use]
    pub fn builder(service_name: impl Into<String>) -> OtelConfigBuilder {
        OtelConfigBuilder::new(service_name.into())
    }

    /// Read every supported env var into a configuration value.
    ///
    /// Honours:
    /// - `OTEL_SERVICE_NAME` / `OTEL_SERVICE_VERSION` /
    ///   `OTEL_SERVICE_INSTANCE_ID` / `OTEL_DEPLOYMENT_ENVIRONMENT`
    /// - `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`
    /// - `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG`
    /// - `OTEL_RESOURCE_ATTRIBUTES`
    ///
    /// Unset variables fall through to the documented defaults; **invalid**
    /// values bail loudly. There are no silent fallbacks.
    ///
    /// # Errors
    /// Returns an error if any env var is malformed:
    /// - non-numeric `OTEL_TRACES_SAMPLER_ARG`
    /// - unknown `OTEL_TRACES_SAMPLER`
    /// - `OTEL_RESOURCE_ATTRIBUTES` / `OTEL_EXPORTER_OTLP_HEADERS` entries
    ///   missing the `=` separator
    pub fn from_env() -> Result<Self> {
        Self::from_env_with(|key| match env::var(key) {
            Ok(raw) => Ok(Some(raw)),
            Err(env::VarError::NotPresent) => Ok(None),
            Err(err) => Err(err).with_context(|| format!("{key} is not valid Unicode")),
        })
    }

    /// Internal hook used by [`Self::from_env`] and tests. The getter is
    /// expected to return `Ok(None)` when the variable is unset and
    /// `Err(_)` only on hard read failures (non-Unicode bytes).
    pub(crate) fn from_env_with<F>(getter: F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Option<String>>,
    {
        let service_name = match optional_value(&getter, "OTEL_SERVICE_NAME")? {
            Some(name) => name,
            None => {
                log::warn!(
                    target: "agent_sdk_otel",
                    "OTEL_SERVICE_NAME is unset; falling back to `{DEFAULT_SERVICE_NAME_FALLBACK}`"
                );
                DEFAULT_SERVICE_NAME_FALLBACK.to_string()
            }
        };
        let service_version = optional_value(&getter, "OTEL_SERVICE_VERSION")?;
        let service_instance_id = optional_value(&getter, "OTEL_SERVICE_INSTANCE_ID")?;
        let deployment_environment = optional_value(&getter, "OTEL_DEPLOYMENT_ENVIRONMENT")?;

        // The OTel spec treats an explicitly empty endpoint as "exporter
        // disabled". `optional_value` strips whitespace and folds empty
        // strings to `None`, so here we go straight to the raw getter to
        // keep that distinction visible.
        let otlp_endpoint = getter("OTEL_EXPORTER_OTLP_ENDPOINT")?;

        let otlp_headers = match optional_value(&getter, "OTEL_EXPORTER_OTLP_HEADERS")? {
            Some(raw) => parse_kv_pairs(&raw).context("OTEL_EXPORTER_OTLP_HEADERS is malformed")?,
            None => Vec::new(),
        };

        let sampler = match optional_value(&getter, "OTEL_TRACES_SAMPLER")? {
            Some(raw) => SamplerKind::from_str(&raw).context("OTEL_TRACES_SAMPLER is malformed")?,
            None => SamplerKind::default(),
        };
        let sample_ratio = match optional_value(&getter, "OTEL_TRACES_SAMPLER_ARG")? {
            Some(raw) => raw
                .parse::<f64>()
                .with_context(|| format!("OTEL_TRACES_SAMPLER_ARG `{raw}` is not a number"))?,
            None => 1.0,
        };

        let additional_resource_attrs = match optional_value(&getter, "OTEL_RESOURCE_ATTRIBUTES")? {
            Some(raw) => parse_kv_pairs(&raw)
                .context("OTEL_RESOURCE_ATTRIBUTES is malformed")?
                .into_iter()
                .map(|(k, v)| KeyValue::new(k, v))
                .collect(),
            None => Vec::new(),
        };

        Ok(Self {
            service_name,
            service_version,
            service_instance_id,
            deployment_environment,
            otlp_endpoint,
            otlp_headers,
            sampler,
            sample_ratio,
            propagated_baggage_keys: Vec::new(),
            capture_payloads: false,
            additional_resource_attrs,
        })
    }

    /// Resolved deployment environment, applying the
    /// `unknown_deployment` default.
    #[must_use]
    pub const fn deployment_environment_or_default(&self) -> &str {
        match &self.deployment_environment {
            Some(value) => value.as_str(),
            None => DEFAULT_DEPLOYMENT_ENVIRONMENT,
        }
    }

    /// Whether the exporter pipeline should be wired up. `Some("")` (an
    /// explicitly empty string) is treated as "disabled" so callers can
    /// switch the exporter off via `OTEL_EXPORTER_OTLP_ENDPOINT=` without
    /// removing the variable.
    #[must_use]
    pub fn exporter_enabled(&self) -> bool {
        match &self.otlp_endpoint {
            Some(value) => !value.trim().is_empty(),
            None => false,
        }
    }

    /// Endpoint string to hand to the OTLP exporter, if enabled.
    #[must_use]
    pub fn endpoint(&self) -> Option<&str> {
        if self.exporter_enabled() {
            self.otlp_endpoint.as_deref()
        } else {
            None
        }
    }
}

impl fmt::Debug for OtelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let masked_headers: Vec<(&str, String)> = self
            .otlp_headers
            .iter()
            .map(|(k, v)| (k.as_str(), format!("<redacted len={}>", v.len())))
            .collect();
        f.debug_struct("OtelConfig")
            .field("service_name", &self.service_name)
            .field("service_version", &self.service_version)
            .field("service_instance_id", &self.service_instance_id)
            .field("deployment_environment", &self.deployment_environment)
            .field("otlp_endpoint", &self.otlp_endpoint)
            .field("otlp_headers", &masked_headers)
            .field("sampler", &self.sampler)
            .field("sample_ratio", &self.sample_ratio)
            .field("propagated_baggage_keys", &self.propagated_baggage_keys)
            .field("capture_payloads", &self.capture_payloads)
            .field(
                "additional_resource_attrs.len",
                &self.additional_resource_attrs.len(),
            )
            .finish()
    }
}

/// Fluent builder for [`OtelConfig`].
pub struct OtelConfigBuilder {
    inner: OtelConfig,
}

impl OtelConfigBuilder {
    fn new(service_name: String) -> Self {
        Self {
            inner: OtelConfig {
                service_name,
                service_version: None,
                service_instance_id: None,
                deployment_environment: None,
                otlp_endpoint: None,
                otlp_headers: Vec::new(),
                sampler: SamplerKind::default(),
                sample_ratio: 1.0,
                propagated_baggage_keys: Vec::new(),
                capture_payloads: false,
                additional_resource_attrs: Vec::new(),
            },
        }
    }

    /// Set `service.version`. Pass `env!("CARGO_PKG_VERSION")` from your
    /// binary to mirror the `OTel` spec default.
    #[must_use]
    pub fn service_version(mut self, version: impl Into<String>) -> Self {
        self.inner.service_version = Some(version.into());
        self
    }

    /// Override `service.instance.id`. When unset the bootstrap helper
    /// generates a fresh UUID v7 per process.
    #[must_use]
    pub fn service_instance_id(mut self, id: impl Into<String>) -> Self {
        self.inner.service_instance_id = Some(id.into());
        self
    }

    /// Set `deployment.environment`. Defaults to `unknown_deployment`.
    #[must_use]
    pub fn deployment_environment(mut self, env: impl Into<String>) -> Self {
        self.inner.deployment_environment = Some(env.into());
        self
    }

    /// Configure the OTLP gRPC endpoint. Pass `None` to disable the
    /// exporter entirely.
    #[must_use]
    pub fn otlp_endpoint(mut self, endpoint: Option<String>) -> Self {
        self.inner.otlp_endpoint = endpoint;
        self
    }

    /// Replace the OTLP header set.
    #[must_use]
    pub fn otlp_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.inner.otlp_headers = headers;
        self
    }

    /// Append a single OTLP header.
    #[must_use]
    pub fn add_otlp_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.inner.otlp_headers.push((key.into(), value.into()));
        self
    }

    /// Pick a [`SamplerKind`].
    #[must_use]
    pub const fn sampler(mut self, sampler: SamplerKind) -> Self {
        self.inner.sampler = sampler;
        self
    }

    /// Sample ratio in `[0.0, 1.0]`, used by ratio-based samplers.
    #[must_use]
    pub const fn sample_ratio(mut self, ratio: f64) -> Self {
        self.inner.sample_ratio = ratio;
        self
    }

    /// Replace the baggage allow-list.
    #[must_use]
    pub fn propagated_baggage_keys(mut self, keys: Vec<String>) -> Self {
        self.inner.propagated_baggage_keys = keys;
        self
    }

    /// Toggle the payload-capture hint surfaced to the rest of the SDK.
    #[must_use]
    pub const fn capture_payloads(mut self, enabled: bool) -> Self {
        self.inner.capture_payloads = enabled;
        self
    }

    /// Append additional `Resource` attributes.
    #[must_use]
    pub fn additional_resource_attrs(mut self, attrs: Vec<KeyValue>) -> Self {
        self.inner.additional_resource_attrs = attrs;
        self
    }

    /// Finish the builder.
    #[must_use]
    pub fn build(self) -> OtelConfig {
        self.inner
    }
}

/// Trim a getter-returned value, folding `Some("")` into `None` so callers
/// don't have to litter their parse paths with empty-string checks.
fn optional_value<F>(getter: &F, key: &str) -> Result<Option<String>>
where
    F: Fn(&str) -> Result<Option<String>>,
{
    let raw = getter(key)?;
    Ok(raw.and_then(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }))
}

/// Parse a `k1=v1,k2=v2` style env value into owned pairs.
///
/// Empty entries (e.g. trailing commas) are skipped silently because they
/// are commonly produced by shell quoting; entries that *contain content*
/// but lack `=` are reported as errors.
fn parse_kv_pairs(raw: &str) -> Result<Vec<(String, String)>> {
    let mut pairs = Vec::new();
    for entry in raw.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((k, v)) = trimmed.split_once('=') else {
            bail!("entry `{trimmed}` is missing `=` separator");
        };
        let key = k.trim();
        if key.is_empty() {
            bail!("entry `{trimmed}` has empty key");
        }
        pairs.push((key.to_string(), v.trim().to_string()));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_round_trip() {
        let cfg = OtelConfig::builder("svc")
            .service_version("1.2.3")
            .service_instance_id("inst-1")
            .deployment_environment("staging")
            .otlp_endpoint(Some("http://otel-collector:4317".to_string()))
            .add_otlp_header("authorization", "Bearer secret")
            .sampler(SamplerKind::ParentBasedTraceIdRatio)
            .sample_ratio(0.25)
            .capture_payloads(true)
            .build();

        assert_eq!(cfg.service_name, "svc");
        assert_eq!(cfg.service_version.as_deref(), Some("1.2.3"));
        assert_eq!(cfg.service_instance_id.as_deref(), Some("inst-1"));
        assert_eq!(cfg.deployment_environment_or_default(), "staging");
        assert!(cfg.exporter_enabled());
        assert_eq!(cfg.endpoint(), Some("http://otel-collector:4317"));
        assert_eq!(cfg.sampler, SamplerKind::ParentBasedTraceIdRatio);
        assert!((cfg.sample_ratio - 0.25).abs() < f64::EPSILON);
        assert!(cfg.capture_payloads);
        assert_eq!(cfg.otlp_headers.len(), 1);
    }

    #[test]
    fn debug_redacts_header_values() {
        let cfg = OtelConfig::builder("svc")
            .add_otlp_header("authorization", "Bearer super-secret-token")
            .build();
        let rendered = format!("{cfg:?}");
        assert!(!rendered.contains("super-secret-token"), "got: {rendered}");
        assert!(rendered.contains("authorization"), "got: {rendered}");
        assert!(rendered.contains("redacted len=25"), "got: {rendered}");
    }

    #[test]
    fn deployment_environment_default_applies() {
        let cfg = OtelConfig::builder("svc").build();
        assert_eq!(
            cfg.deployment_environment_or_default(),
            "unknown_deployment"
        );
    }

    #[test]
    fn empty_endpoint_disables_exporter() {
        let cfg = OtelConfig::builder("svc")
            .otlp_endpoint(Some(String::new()))
            .build();
        assert!(!cfg.exporter_enabled());
        assert_eq!(cfg.endpoint(), None);
    }

    #[test]
    fn parse_kv_pairs_handles_trim_and_empties() -> Result<()> {
        let pairs = parse_kv_pairs("a=1, b = 2 ,,c=hello world")?;
        assert_eq!(
            pairs,
            vec![
                ("a".to_string(), "1".to_string()),
                ("b".to_string(), "2".to_string()),
                ("c".to_string(), "hello world".to_string()),
            ]
        );
        Ok(())
    }

    #[test]
    fn parse_kv_pairs_rejects_missing_equals() {
        let Err(err) = parse_kv_pairs("a=1,malformed") else {
            panic!("must error");
        };
        assert!(format!("{err}").contains("missing `=`"), "got: {err}");
    }

    #[test]
    fn parse_kv_pairs_rejects_empty_key() {
        let Err(err) = parse_kv_pairs("=value") else {
            panic!("must error");
        };
        assert!(format!("{err}").contains("empty key"), "got: {err}");
    }

    /// Build a getter closure backed by an in-memory `Vec`, useful
    /// for exercising `from_env_with` without touching the process
    /// environment (which would require `unsafe` under Rust 2024).
    fn fake_getter(
        entries: Vec<(&'static str, &'static str)>,
    ) -> impl Fn(&str) -> Result<Option<String>> {
        let owned: Vec<(String, String)> = entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key: &str| {
            for (k, v) in &owned {
                if k == key {
                    return Ok(Some(v.clone()));
                }
            }
            Ok(None)
        }
    }

    #[test]
    fn from_env_with_default_values_when_unset() -> Result<()> {
        let cfg = OtelConfig::from_env_with(|_| Ok(None))?;
        assert_eq!(cfg.service_name, "agent-sdk");
        assert!(cfg.service_version.is_none());
        assert_eq!(cfg.sampler, SamplerKind::ParentBasedTraceIdRatio);
        assert!((cfg.sample_ratio - 1.0).abs() < f64::EPSILON);
        assert!(cfg.otlp_endpoint.is_none());
        assert!(cfg.otlp_headers.is_empty());
        assert!(!cfg.exporter_enabled());
        Ok(())
    }

    #[test]
    fn from_env_with_propagates_known_values() -> Result<()> {
        let getter = fake_getter(vec![
            ("OTEL_SERVICE_NAME", "svc"),
            ("OTEL_SERVICE_VERSION", "1.2.3"),
            ("OTEL_DEPLOYMENT_ENVIRONMENT", "prod"),
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"),
            ("OTEL_EXPORTER_OTLP_HEADERS", "x-tenant=alpha,x-flag=true"),
            ("OTEL_TRACES_SAMPLER", "parentbased_traceidratio"),
            ("OTEL_TRACES_SAMPLER_ARG", "0.05"),
            ("OTEL_RESOURCE_ATTRIBUTES", "service.namespace=demo"),
        ]);
        let cfg = OtelConfig::from_env_with(getter)?;
        assert_eq!(cfg.service_name, "svc");
        assert_eq!(cfg.service_version.as_deref(), Some("1.2.3"));
        assert_eq!(cfg.deployment_environment.as_deref(), Some("prod"));
        assert_eq!(cfg.endpoint(), Some("http://collector:4317"));
        assert_eq!(cfg.otlp_headers.len(), 2);
        assert_eq!(cfg.sampler, SamplerKind::ParentBasedTraceIdRatio);
        assert!((cfg.sample_ratio - 0.05).abs() < f64::EPSILON);
        assert_eq!(cfg.additional_resource_attrs.len(), 1);
        Ok(())
    }

    #[test]
    fn from_env_with_rejects_non_numeric_sampler_arg() {
        let getter = fake_getter(vec![("OTEL_TRACES_SAMPLER_ARG", "not-a-number")]);
        let Err(err) = OtelConfig::from_env_with(getter) else {
            panic!("must reject non-numeric sampler arg");
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("OTEL_TRACES_SAMPLER_ARG") && msg.contains("not a number"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn from_env_with_rejects_unknown_sampler() {
        let getter = fake_getter(vec![("OTEL_TRACES_SAMPLER", "definitely-not-a-sampler")]);
        let Err(err) = OtelConfig::from_env_with(getter) else {
            panic!("must reject unknown sampler");
        };
        let chained = format!("{err:#}");
        assert!(
            chained.contains("OTEL_TRACES_SAMPLER is malformed"),
            "unexpected error: {chained}"
        );
    }

    #[test]
    fn from_env_with_rejects_malformed_headers() {
        let getter = fake_getter(vec![("OTEL_EXPORTER_OTLP_HEADERS", "bad-header-no-equals")]);
        let Err(err) = OtelConfig::from_env_with(getter) else {
            panic!("must reject malformed headers");
        };
        let chained = format!("{err:#}");
        assert!(
            chained.contains("OTEL_EXPORTER_OTLP_HEADERS is malformed"),
            "unexpected error: {chained}"
        );
    }

    #[test]
    fn from_env_with_treats_empty_endpoint_as_disabled() -> Result<()> {
        let getter = fake_getter(vec![("OTEL_EXPORTER_OTLP_ENDPOINT", "")]);
        let cfg = OtelConfig::from_env_with(getter)?;
        assert_eq!(cfg.otlp_endpoint.as_deref(), Some(""));
        assert!(!cfg.exporter_enabled());
        assert_eq!(cfg.endpoint(), None);
        Ok(())
    }
}
