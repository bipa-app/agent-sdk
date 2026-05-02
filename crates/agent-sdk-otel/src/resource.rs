//! Resource construction for the bootstrap helper.
//!
//! Builds an [`opentelemetry_sdk::Resource`] populated with the SDK's
//! recommended defaults plus any caller-supplied attributes. Defaults are
//! applied without consulting the `OTel` `EnvResourceDetector` because
//! every env var is already routed through
//! [`crate::config::OtelConfig::from_env`] — keeping a single source of
//! truth makes the resulting resource deterministic for tests and trivial
//! to reason about.

// Same rationale as `config.rs`: clippy's `option_if_let_else` lint
// pushes us toward `map_or_else`, which the project rule forbids.
#![allow(clippy::option_if_let_else)]

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use uuid::Uuid;

use crate::config::OtelConfig;

/// `telemetry.sdk.name` reported on every emitted span/metric.
const TELEMETRY_SDK_NAME: &str = "agent-sdk";
/// `telemetry.sdk.version` — the version of the bootstrap crate itself.
const TELEMETRY_SDK_VERSION: &str = env!("CARGO_PKG_VERSION");
/// `OpenTelemetry` semantic-convention schema URL applied to the resource.
const SCHEMA_URL: &str = "https://opentelemetry.io/schemas/1.30.0";

/// Build the `Resource` for the configured service.
///
/// Sets at minimum:
/// - `service.name`
/// - `service.version` (only when `cfg.service_version` is set — callers
///   that want to mirror the `OTel` spec default should populate this from
///   the binary's `CARGO_PKG_VERSION`)
/// - `service.instance.id` (`cfg.service_instance_id` or a fresh UUID v7
///   per process)
/// - `deployment.environment` (`cfg.deployment_environment` or the
///   `unknown_deployment` default)
/// - `telemetry.sdk.name = agent-sdk`
/// - `telemetry.sdk.version` of the bootstrap crate
/// - any `cfg.additional_resource_attrs`
pub fn build(cfg: &OtelConfig) -> Resource {
    let service_instance_id = match &cfg.service_instance_id {
        Some(id) => id.clone(),
        None => Uuid::now_v7().to_string(),
    };
    let deployment_environment = cfg.deployment_environment_or_default().to_string();

    let mut attrs = vec![
        KeyValue::new("service.instance.id", service_instance_id),
        KeyValue::new("deployment.environment", deployment_environment),
        KeyValue::new("telemetry.sdk.name", TELEMETRY_SDK_NAME),
        KeyValue::new("telemetry.sdk.version", TELEMETRY_SDK_VERSION),
    ];
    if let Some(version) = &cfg.service_version {
        attrs.push(KeyValue::new("service.version", version.clone()));
    }
    attrs.extend(cfg.additional_resource_attrs.iter().cloned());

    Resource::builder_empty()
        .with_service_name(cfg.service_name.clone())
        .with_attributes(attrs)
        .with_schema_url(Vec::<KeyValue>::new(), SCHEMA_URL)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::Key;

    fn lookup(resource: &Resource, key: &str) -> Option<String> {
        resource
            .get(&Key::new(key.to_string()))
            .map(|v| v.to_string())
    }

    #[test]
    fn defaults_applied_when_unset() -> anyhow::Result<()> {
        let cfg = OtelConfig::builder("svc-default").build();
        let resource = build(&cfg);

        assert_eq!(
            lookup(&resource, "service.name").as_deref(),
            Some("svc-default")
        );
        assert_eq!(
            lookup(&resource, "deployment.environment").as_deref(),
            Some("unknown_deployment")
        );
        assert_eq!(
            lookup(&resource, "telemetry.sdk.name").as_deref(),
            Some(TELEMETRY_SDK_NAME)
        );
        assert_eq!(
            lookup(&resource, "telemetry.sdk.version").as_deref(),
            Some(TELEMETRY_SDK_VERSION)
        );
        // Instance id is auto-generated when missing.
        let instance_id = lookup(&resource, "service.instance.id")
            .ok_or_else(|| anyhow::anyhow!("service.instance.id missing"))?;
        assert!(!instance_id.is_empty());
        // service.version is not set unless caller-supplied.
        assert!(
            lookup(&resource, "service.version").is_none(),
            "service.version should be absent when not configured"
        );
        Ok(())
    }

    #[test]
    fn explicit_overrides_used() {
        let cfg = OtelConfig::builder("svc")
            .service_version("9.9.9")
            .service_instance_id("inst-fixed")
            .deployment_environment("prod")
            .additional_resource_attrs(vec![KeyValue::new("custom.tag", "true")])
            .build();
        let resource = build(&cfg);

        assert_eq!(
            lookup(&resource, "service.version").as_deref(),
            Some("9.9.9")
        );
        assert_eq!(
            lookup(&resource, "service.instance.id").as_deref(),
            Some("inst-fixed")
        );
        assert_eq!(
            lookup(&resource, "deployment.environment").as_deref(),
            Some("prod")
        );
        assert_eq!(lookup(&resource, "custom.tag").as_deref(), Some("true"));
    }
}
