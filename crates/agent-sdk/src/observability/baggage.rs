//! Baggage helpers for propagating trace-level attributes onto every span.
//!
//! Langfuse (and any other backend that filters by user/session) requires
//! the trace-level attributes (`user.id`, `session.id`, `langfuse.user.id`,
//! `langfuse.session.id`, `deployment.environment`) to appear on **every**
//! span, not just the root, so that filters and aggregations work.
//!
//! The `with_*` helpers attach the canonical baggage key to a [`Context`]
//! while preserving any existing baggage entries. The
//! [`copy_baggage_to_active_span`] helper is called from every SDK span
//! creation site so the allow-listed keys land on the span as attributes.
//!
//! Outbound propagation filtering (i.e. the allow-list applied to the
//! W3C Baggage header) lives in C3 and is intentionally not handled here.

use opentelemetry::baggage::{Baggage, BaggageExt, KeyValueMetadata};
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::Span;
use opentelemetry::{Context, KeyValue};

use super::attrs;

/// Baggage key for the human (or service account) running the agent.
pub const BAGGAGE_USER_ID: &str = "user.id";

/// Baggage key for the session/conversation grouping.
pub const BAGGAGE_SESSION_ID: &str = "session.id";

/// Langfuse-specific baggage key for `userId` filters.
pub const BAGGAGE_LANGFUSE_USER_ID: &str = "langfuse.user.id";

/// Langfuse-specific baggage key for `sessionId` filters.
pub const BAGGAGE_LANGFUSE_SESSION_ID: &str = "langfuse.session.id";

/// Baggage key for the deployment environment (`prod`, `staging`, …).
pub const BAGGAGE_DEPLOYMENT_ENVIRONMENT: &str = "deployment.environment";

/// Allow-list of baggage keys copied to every SDK-emitted span.
///
/// Stays small on purpose. Adding new keys means widening the contract
/// reviewers verify against the inventory in `PHASE_9_INVENTORY.md`.
const PROPAGATED_KEYS: &[&str] = &[
    BAGGAGE_USER_ID,
    BAGGAGE_SESSION_ID,
    BAGGAGE_LANGFUSE_USER_ID,
    BAGGAGE_LANGFUSE_SESSION_ID,
    BAGGAGE_DEPLOYMENT_ENVIRONMENT,
];

/// Returns the context with `user.id` set in baggage, preserving existing entries.
#[must_use]
pub fn with_user_id(cx: &Context, user_id: impl Into<String>) -> Context {
    with_attribute(cx, BAGGAGE_USER_ID, user_id.into())
}

/// Returns the context with `session.id` set in baggage, preserving existing entries.
#[must_use]
pub fn with_session_id(cx: &Context, session_id: impl Into<String>) -> Context {
    with_attribute(cx, BAGGAGE_SESSION_ID, session_id.into())
}

/// Returns the context with the supplied attributes added to baggage,
/// preserving every existing entry.
#[must_use]
pub fn with_attributes(cx: &Context, kvs: impl IntoIterator<Item = KeyValue>) -> Context {
    let baggage: Baggage = cx
        .baggage()
        .iter()
        .map(|(key, (value, metadata))| {
            KeyValueMetadata::new(key.clone(), value.clone(), metadata.clone())
        })
        .chain(
            kvs.into_iter()
                .map(|kv| KeyValueMetadata::new(kv.key, kv.value, "")),
        )
        .collect();
    cx.with_baggage(baggage)
}

fn with_attribute(cx: &Context, key: &'static str, value: String) -> Context {
    with_attributes(cx, std::iter::once(KeyValue::new(key, value)))
}

/// Copy every allow-listed baggage entry from the current context onto
/// the supplied span as attributes.
///
/// Skips silently when:
/// * the span is not recording (sampling decision was "drop"); or
/// * a given baggage key is absent (no empty-string defaults).
///
/// `session.id` is additionally mirrored onto `gen_ai.conversation.id`
/// so downstream tools that filter by the canonical `OTel` `GenAI` key
/// keep working.
pub fn copy_baggage_to_active_span(span: &mut BoxedSpan) {
    if !span.is_recording() {
        return;
    }

    let cx = Context::current();
    let baggage = cx.baggage();

    for &key in PROPAGATED_KEYS {
        if let Some(value) = baggage.get(key) {
            let value_string = value.as_str().to_string();
            if key == BAGGAGE_SESSION_ID {
                span.set_attribute(KeyValue::new(
                    attrs::GEN_AI_CONVERSATION_ID,
                    value_string.clone(),
                ));
            }
            span.set_attribute(KeyValue::new(key, value_string));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_user_id_preserves_existing_baggage() {
        let cx = Context::current_with_baggage([KeyValue::new("trace.tag", "v1")]);
        let cx = with_user_id(&cx, "user-42");

        assert_eq!(
            cx.baggage().get("trace.tag").map(ToString::to_string),
            Some("v1".to_string())
        );
        assert_eq!(
            cx.baggage().get(BAGGAGE_USER_ID).map(ToString::to_string),
            Some("user-42".to_string())
        );
    }

    #[test]
    fn with_session_id_overwrites_previous_value() {
        let cx = Context::current_with_baggage([KeyValue::new(BAGGAGE_SESSION_ID, "old")]);
        let cx = with_session_id(&cx, "new");

        assert_eq!(
            cx.baggage()
                .get(BAGGAGE_SESSION_ID)
                .map(ToString::to_string),
            Some("new".to_string())
        );
    }

    #[test]
    fn with_attributes_merges_multiple_keys() {
        let cx = Context::current_with_baggage([KeyValue::new("trace.tag", "v1")]);
        let cx = with_attributes(
            &cx,
            [
                KeyValue::new(BAGGAGE_USER_ID, "alice"),
                KeyValue::new(BAGGAGE_LANGFUSE_USER_ID, "alice"),
            ],
        );

        assert_eq!(
            cx.baggage().get("trace.tag").map(ToString::to_string),
            Some("v1".to_string())
        );
        assert_eq!(
            cx.baggage().get(BAGGAGE_USER_ID).map(ToString::to_string),
            Some("alice".to_string())
        );
        assert_eq!(
            cx.baggage()
                .get(BAGGAGE_LANGFUSE_USER_ID)
                .map(ToString::to_string),
            Some("alice".to_string())
        );
    }

    #[test]
    fn propagated_keys_constant_includes_all_advertised_constants() {
        assert!(PROPAGATED_KEYS.contains(&BAGGAGE_USER_ID));
        assert!(PROPAGATED_KEYS.contains(&BAGGAGE_SESSION_ID));
        assert!(PROPAGATED_KEYS.contains(&BAGGAGE_LANGFUSE_USER_ID));
        assert!(PROPAGATED_KEYS.contains(&BAGGAGE_LANGFUSE_SESSION_ID));
        assert!(PROPAGATED_KEYS.contains(&BAGGAGE_DEPLOYMENT_ENVIRONMENT));
    }
}
