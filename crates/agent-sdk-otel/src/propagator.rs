//! Allow-list wrapper around the upstream `BaggagePropagator`.
//!
//! `OTel` baggage is propagated to downstream services and third-party
//! APIs (HTTP/gRPC headers). [Langfuse explicitly warns][lf]:
//!
//! > Do not include sensitive information (passwords, API keys, personal
//! > data, etc.) in baggage when using this approach, as it will be
//! > transmitted to all downstream services.
//!
//! [`AllowListBaggagePropagator`] enforces an exact-match allow-list on
//! **outbound** baggage serialization. Inbound entries are preserved
//! unchanged so we can still consume whatever upstream services hand us;
//! we just don't forward anything we didn't explicitly approve.
//!
//! [lf]: https://langfuse.com/integrations/native/opentelemetry#propagating-attributes

use std::collections::HashSet;

use opentelemetry::Context;
use opentelemetry::baggage::{Baggage, BaggageExt, KeyValueMetadata};
use opentelemetry::propagation::text_map_propagator::FieldIter;
use opentelemetry::propagation::{Extractor, Injector, TextMapPropagator};
use opentelemetry_sdk::propagation::BaggagePropagator;

/// Baseline baggage key for `user.id`.
///
/// Mirrors the constants in `agent_sdk::observability::baggage` (Track
/// A3). The duplication is intentional so this propagator has no
/// runtime dependency on `agent-sdk` beyond the existing C2 capture
/// gate hook in [`crate::install_global_provider`].
pub const BASELINE_USER_ID: &str = "user.id";
/// Baseline baggage key for `session.id`.
pub const BASELINE_SESSION_ID: &str = "session.id";
/// Baseline baggage key for Langfuse-specific `userId` filtering.
pub const BASELINE_LANGFUSE_USER_ID: &str = "langfuse.user.id";
/// Baseline baggage key for Langfuse-specific `sessionId` filtering.
pub const BASELINE_LANGFUSE_SESSION_ID: &str = "langfuse.session.id";
/// Baseline baggage key for `deployment.environment`.
pub const BASELINE_DEPLOYMENT_ENVIRONMENT: &str = "deployment.environment";

/// W3C Baggage propagator that filters outbound entries through an
/// exact-match allow-list of keys.
///
/// `inject_context` rebuilds the outgoing [`Baggage`] containing only
/// the allow-listed entries before delegating to the inner upstream
/// propagator.  `extract_with_context` and `fields` delegate untouched
/// — inbound services may still hand us anything they want, we just
/// don't forward it onwards.
///
/// # Hard rule
///
/// Allow-list comparison is **exact match only**. Adding `"user"` to
/// the allow-list does **not** propagate `"user.id"`. Prefix or
/// wildcard matching requires an explicit policy decision.
#[derive(Debug)]
pub struct AllowListBaggagePropagator {
    inner: BaggagePropagator,
    allow: HashSet<String>,
}

impl AllowListBaggagePropagator {
    /// Construct a propagator that forwards only the supplied keys.
    ///
    /// Empty allow-lists are honoured: the resulting propagator will
    /// never inject a `baggage` header.
    #[must_use]
    pub fn new(allow: impl IntoIterator<Item = String>) -> Self {
        Self {
            inner: BaggagePropagator::new(),
            allow: allow.into_iter().collect(),
        }
    }

    /// The five baseline keys the SDK considers safe to propagate by
    /// default.
    ///
    /// Mirrors `agent_sdk::observability::baggage::PROPAGATED_KEYS`
    /// (Track A3). Keep these two lists in lockstep when widening the
    /// contract — see `PHASE_9_INVENTORY.md`.
    #[must_use]
    pub fn baseline_allow_list() -> Vec<String> {
        vec![
            BASELINE_USER_ID.to_owned(),
            BASELINE_SESSION_ID.to_owned(),
            BASELINE_LANGFUSE_USER_ID.to_owned(),
            BASELINE_LANGFUSE_SESSION_ID.to_owned(),
            BASELINE_DEPLOYMENT_ENVIRONMENT.to_owned(),
        ]
    }

    /// Build a [`Baggage`] containing only the entries whose keys are
    /// on the allow-list.
    fn filter(&self, cx: &Context) -> Baggage {
        cx.baggage()
            .iter()
            .filter(|(key, _)| self.allow.contains(key.as_str()))
            .map(|(key, (value, metadata))| {
                KeyValueMetadata::new(key.clone(), value.clone(), metadata.clone())
            })
            .collect()
    }
}

impl TextMapPropagator for AllowListBaggagePropagator {
    fn inject_context(&self, cx: &Context, injector: &mut dyn Injector) {
        let filtered = self.filter(cx);
        if filtered.is_empty() {
            // Mirror the upstream `BaggagePropagator` behaviour: never
            // emit a `baggage` header for an empty bag.
            return;
        }
        let filtered_cx = cx.with_baggage(filtered);
        self.inner.inject_context(&filtered_cx, injector);
    }

    fn extract_with_context(&self, cx: &Context, extractor: &dyn Extractor) -> Context {
        // Inbound values are preserved as-is — we only sanitise on the
        // way out. Whatever upstream hands us is still visible inside
        // the process; the next outbound call drops anything that
        // isn't allow-listed.
        self.inner.extract_with_context(cx, extractor)
    }

    fn fields(&self) -> FieldIter<'_> {
        self.inner.fields()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::KeyValue;
    use opentelemetry::baggage::BaggageExt;
    use std::collections::HashMap;

    fn baseline() -> AllowListBaggagePropagator {
        AllowListBaggagePropagator::new(AllowListBaggagePropagator::baseline_allow_list())
    }

    #[test]
    fn baseline_allow_list_contains_exactly_five_keys() {
        let baseline = AllowListBaggagePropagator::baseline_allow_list();
        assert_eq!(baseline.len(), 5);
        assert!(baseline.iter().any(|k| k == BASELINE_USER_ID));
        assert!(baseline.iter().any(|k| k == BASELINE_SESSION_ID));
        assert!(baseline.iter().any(|k| k == BASELINE_LANGFUSE_USER_ID));
        assert!(baseline.iter().any(|k| k == BASELINE_LANGFUSE_SESSION_ID));
        assert!(
            baseline
                .iter()
                .any(|k| k == BASELINE_DEPLOYMENT_ENVIRONMENT)
        );
    }

    #[test]
    fn inject_drops_non_allow_listed_keys() {
        let propagator = baseline();
        let cx = Context::current_with_baggage([
            KeyValue::new("user.id", "alice"),
            KeyValue::new("password", "hunter2"),
            KeyValue::new("api_key", "shhh"),
        ]);
        let mut headers: HashMap<String, String> = HashMap::new();
        propagator.inject_context(&cx, &mut headers);

        let Some(header) = headers.get("baggage") else {
            panic!("allow-listed user.id should produce a baggage header; got: {headers:?}");
        };
        assert!(header.contains("user.id=alice"), "got: {header}");
        assert!(!header.contains("password"), "got: {header}");
        assert!(!header.contains("hunter2"), "got: {header}");
        assert!(!header.contains("api_key"), "got: {header}");
    }

    #[test]
    fn inject_skips_header_when_no_keys_pass_filter() {
        let propagator = baseline();
        let cx = Context::current_with_baggage([KeyValue::new("password", "hunter2")]);
        let mut headers: HashMap<String, String> = HashMap::new();
        propagator.inject_context(&cx, &mut headers);
        assert!(!headers.contains_key("baggage"), "got: {headers:?}");
    }

    #[test]
    fn extension_propagates_extra_keys() {
        let extra =
            AllowListBaggagePropagator::new(["user.id".to_owned(), "request.id".to_owned()]);
        let cx = Context::current_with_baggage([
            KeyValue::new("user.id", "alice"),
            KeyValue::new("request.id", "req-42"),
            KeyValue::new("password", "hunter2"),
        ]);
        let mut headers: HashMap<String, String> = HashMap::new();
        extra.inject_context(&cx, &mut headers);
        let Some(header) = headers.get("baggage") else {
            panic!("baggage header should be injected; got: {headers:?}");
        };
        assert!(header.contains("user.id=alice"), "got: {header}");
        assert!(header.contains("request.id=req-42"), "got: {header}");
        assert!(!header.contains("password"), "got: {header}");
    }

    #[test]
    fn extract_preserves_inbound_baggage_unchanged() {
        let propagator = baseline();
        let mut headers: HashMap<String, String> = HashMap::new();
        // Inbound carries something the allow-list would normally
        // reject — `extract_with_context` must still surface it.
        headers.insert(
            "baggage".to_owned(),
            "user.id=alice,password=hunter2".to_owned(),
        );
        let cx = propagator.extract(&headers);
        let baggage = cx.baggage();
        assert_eq!(
            baggage.get("user.id").map(ToString::to_string),
            Some("alice".into())
        );
        assert_eq!(
            baggage.get("password").map(ToString::to_string),
            Some("hunter2".into())
        );
    }

    #[test]
    fn allow_list_uses_exact_match_only() {
        // Adding `"user"` to the allow-list must not propagate
        // `"user.id"` — exact match, no prefix matching.
        let propagator = AllowListBaggagePropagator::new(["user".to_owned()]);
        let cx = Context::current_with_baggage([
            KeyValue::new("user.id", "alice"),
            KeyValue::new("user", "bob"),
        ]);
        let mut headers: HashMap<String, String> = HashMap::new();
        propagator.inject_context(&cx, &mut headers);
        let Some(header) = headers.get("baggage") else {
            panic!("baggage header should be injected; got: {headers:?}");
        };
        assert!(header.contains("user=bob"), "got: {header}");
        assert!(!header.contains("user.id"), "got: {header}");
        assert!(!header.contains("alice"), "got: {header}");
    }

    #[test]
    fn empty_baggage_yields_no_header() {
        let propagator = baseline();
        let cx = Context::current();
        let mut headers: HashMap<String, String> = HashMap::new();
        propagator.inject_context(&cx, &mut headers);
        assert!(headers.is_empty(), "got: {headers:?}");
    }

    #[test]
    fn fields_advertises_baggage_header() {
        let propagator = baseline();
        let fields: Vec<&str> = propagator.fields().collect();
        assert!(fields.contains(&"baggage"), "got: {fields:?}");
    }
}
