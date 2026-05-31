//! W3C trace-context propagation across a tonic gRPC boundary.
//!
//! These adapters let a gRPC **client** inject the current trace context
//! into outgoing call metadata and a gRPC **server** extract it to
//! continue the distributed trace — the same pattern Bipa's main gRPC
//! server and mobile apps use, built on the globally-installed
//! [`TraceContextPropagator`] (see [`crate::install_global_provider`]).
//!
//! # How to propagate trace context to the bip daemon
//!
//! The daemon already **extracts** on the server side: its
//! `submit_thread_work` handler reads the inbound `traceparent` from the
//! gRPC call metadata and persists it on the root turn, so the daemon's
//! `invoke_agent` span (and every `chat` / `execute_tool` span under it)
//! continues the caller's trace. A frontend only has to **inject** on the
//! client side.
//!
//! ## Client: inject on every outgoing call
//!
//! Attach [`trace_context_interceptor`] when constructing the tonic
//! client, exactly like Bipa-iOS's `OpenTelemetryInterceptor`:
//!
//! ```no_run
//! # use tonic::transport::Channel;
//! # fn example(channel: Channel) {
//! // Whatever span is current when a call is made becomes the parent of
//! // the daemon's `invoke_agent` span.
//! let interceptor = agent_sdk_otel::grpc::trace_context_interceptor();
//! // let mut client = MyServiceClient::with_interceptor(channel, interceptor);
//! # let _ = (channel, interceptor);
//! # }
//! ```
//!
//! Start a client span around the user-visible operation so the injected
//! `traceparent` points at something meaningful:
//!
//! ```no_run
//! # use opentelemetry::trace::{Tracer, TraceContextExt};
//! # use opentelemetry::{global, Context};
//! let tracer = global::tracer("my-frontend");
//! let span = tracer.start("submit_prompt");
//! let _guard = Context::current_with_span(span).attach();
//! // ... make the gRPC call here; the interceptor injects this span ...
//! ```
//!
//! ## Server: extract to continue the trace
//!
//! A server handler / tower layer continues the trace with
//! [`extract_context`]:
//!
//! ```no_run
//! # use tonic::metadata::MetadataMap;
//! # use opentelemetry::trace::{Tracer, TraceContextExt};
//! # use opentelemetry::global;
//! # fn handle(metadata: &MetadataMap) {
//! let parent = agent_sdk_otel::grpc::extract_context(metadata);
//! let tracer = global::tracer("my-service");
//! let span = tracer.start_with_context("handle_rpc", &parent);
//! # let _ = span;
//! # }
//! ```

use opentelemetry::Context;
use opentelemetry::global;
use opentelemetry::propagation::{Extractor, Injector};
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};

/// Adapts a tonic [`MetadataMap`] as an `OTel` [`Injector`] so a
/// [`TextMapPropagator`](opentelemetry::propagation::TextMapPropagator)
/// can write `traceparent` / `tracestate` / `baggage` into outgoing gRPC
/// metadata. Non-ASCII keys / values are silently skipped (W3C headers
/// are always ASCII).
pub struct MetadataInjector<'a>(pub &'a mut MetadataMap);

impl Injector for MetadataInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(key), Ok(value)) = (
            MetadataKey::from_bytes(key.as_bytes()),
            MetadataValue::try_from(&value),
        ) {
            self.0.insert(key, value);
        }
    }
}

/// Adapts a tonic [`MetadataMap`] as an `OTel` [`Extractor`] so a
/// propagator can read the trace context out of inbound gRPC metadata.
pub struct MetadataExtractor<'a>(pub &'a MetadataMap);

impl Extractor for MetadataExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0
            .iter()
            .filter_map(|entry| match entry {
                tonic::metadata::KeyAndValueRef::Ascii(key, _) => Some(key.as_str()),
                tonic::metadata::KeyAndValueRef::Binary(_, _) => None,
            })
            .collect()
    }
}

/// Inject `cx`'s trace context into outgoing gRPC `metadata` using the
/// globally-installed propagator. No-op when no provider is installed.
pub fn inject_context(cx: &Context, metadata: &mut MetadataMap) {
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(cx, &mut MetadataInjector(metadata));
    });
}

/// Extract the parent [`Context`] from inbound gRPC `metadata`.
///
/// Returns [`Context::current`] augmented with the extracted remote span
/// context + baggage. When the caller propagated nothing, the result
/// carries no valid span (callers can check
/// `cx.span().span_context().is_valid()`).
#[must_use]
pub fn extract_context(metadata: &MetadataMap) -> Context {
    global::get_text_map_propagator(|propagator| propagator.extract(&MetadataExtractor(metadata)))
}

/// A tonic client interceptor that injects the **current** OTel context
/// (`traceparent` + allow-listed baggage) into every outgoing request.
///
/// Pass to `Client::with_interceptor(channel, trace_context_interceptor())`.
/// Cloneable + `FnMut` so it satisfies tonic's `Interceptor` bound and can
/// be shared across cloned clients.
pub fn trace_context_interceptor()
-> impl FnMut(tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status> + Clone {
    |mut request: tonic::Request<()>| {
        inject_context(&Context::current(), request.metadata_mut());
        Ok(request)
    }
}
