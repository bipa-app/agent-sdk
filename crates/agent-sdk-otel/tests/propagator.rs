//! Integration tests for the global baggage allow-list.
//!
//! Drives [`agent_sdk_otel::install_global_provider`] end-to-end and
//! exercises the resulting global propagator against a `HashMap`
//! injector to confirm the wrapper actually reaches the global state.
//! The unit-level coverage of [`agent_sdk_otel::AllowListBaggagePropagator`]
//! itself lives in the propagator module's `#[cfg(test)] mod tests`.

use agent_sdk_otel::{OtelConfig, SamplerKind, install_global_provider};
use anyhow::{Context, Result};
use opentelemetry::KeyValue;
use opentelemetry::baggage::BaggageExt;
use opentelemetry::context::Context as OtelContext;
use opentelemetry::global;
use std::collections::HashMap;
use std::sync::Mutex;

/// Globally serialise tests that mutate the `OTel` global propagator —
/// `install_global_provider` overwrites process-wide state, so running
/// concurrently produces flaky behaviour. The smoke tests in
/// `tests/install.rs` follow the same pattern.
static GLOBAL: Mutex<()> = Mutex::new(());

fn install_with_keys(keys: Vec<String>) -> Result<agent_sdk_otel::OtelGuard> {
    let cfg = OtelConfig::builder("agent-sdk-otel-c3-test")
        .otlp_endpoint(None)
        .sampler(SamplerKind::AlwaysOff)
        .propagated_baggage_keys(keys)
        .build();
    install_global_provider(&cfg)
}

fn inject_via_global(cx: &OtelContext) -> HashMap<String, String> {
    let mut headers: HashMap<String, String> = HashMap::new();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(cx, &mut headers);
    });
    headers
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn default_allow_list_drops_non_baseline_keys_through_global_propagator() -> Result<()> {
    let _g = GLOBAL.lock().ok().context("test mutex poisoned")?;

    // Empty `propagated_baggage_keys` falls back to the SDK baseline
    // (the five keys defined in `agent_sdk::observability::baggage`).
    let guard = install_with_keys(Vec::new())?;

    let cx = OtelContext::current_with_baggage([
        KeyValue::new("user.id", "alice"),
        KeyValue::new("session.id", "sess-7"),
        KeyValue::new("password", "hunter2"),
        KeyValue::new("api_key", "shhh"),
    ]);
    let headers = inject_via_global(&cx);

    let Some(header) = headers.get("baggage") else {
        panic!("baseline allow-list should propagate user.id and session.id; got: {headers:?}");
    };
    assert!(header.contains("user.id=alice"), "got: {header}");
    assert!(header.contains("session.id=sess-7"), "got: {header}");
    assert!(!header.contains("password"), "got: {header}");
    assert!(!header.contains("hunter2"), "got: {header}");
    assert!(!header.contains("api_key"), "got: {header}");

    guard.shutdown()?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn custom_allow_list_propagates_only_configured_keys() -> Result<()> {
    let _g = GLOBAL.lock().ok().context("test mutex poisoned")?;

    let guard = install_with_keys(vec!["user.id".to_owned(), "request.id".to_owned()])?;

    let cx = OtelContext::current_with_baggage([
        KeyValue::new("user.id", "alice"),
        KeyValue::new("request.id", "req-42"),
        // Baseline key that's not in the custom list — must be dropped.
        KeyValue::new("session.id", "sess-7"),
        // Sensitive key — must always be dropped.
        KeyValue::new("password", "hunter2"),
    ]);
    let headers = inject_via_global(&cx);

    let Some(header) = headers.get("baggage") else {
        panic!("custom allow-list should propagate user.id and request.id; got: {headers:?}");
    };
    assert!(header.contains("user.id=alice"), "got: {header}");
    assert!(header.contains("request.id=req-42"), "got: {header}");
    assert!(!header.contains("session.id"), "got: {header}");
    assert!(!header.contains("password"), "got: {header}");

    guard.shutdown()?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn extract_via_global_preserves_non_allow_listed_inbound_baggage() -> Result<()> {
    let _g = GLOBAL.lock().ok().context("test mutex poisoned")?;

    let guard = install_with_keys(Vec::new())?;

    // Inbound carries something the allow-list would normally reject —
    // the global propagator must still surface it after extract.
    let mut headers: HashMap<String, String> = HashMap::new();
    headers.insert(
        "baggage".to_owned(),
        "user.id=alice,password=hunter2".to_owned(),
    );

    let cx = global::get_text_map_propagator(|propagator| propagator.extract(&headers));
    let baggage = cx.baggage();
    assert_eq!(
        baggage.get("user.id").map(ToString::to_string),
        Some("alice".into())
    );
    assert_eq!(
        baggage.get("password").map(ToString::to_string),
        Some("hunter2".into())
    );

    guard.shutdown()?;
    Ok(())
}
