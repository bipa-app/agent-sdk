//! Authoritative tool audit sink.
//!
//! The [`ToolAuditSink`] trait is the server-side audit surface for
//! tool lifecycle outcomes. A single sink receives one
//! [`ToolAuditRecord`] per lifecycle transition — blocked,
//! `requires-confirmation`, cached, replayed, invalidated, completed,
//! and `persistence-failed` — and is free to forward them to any
//! durable audit backend.
//!
//! # Why this exists
//!
//! Before Phase 1.6 the only audit surface on the authoritative path
//! was [`AgentHooks::post_tool_use`](crate::hooks::AgentHooks::post_tool_use),
//! which fires exactly once, only after successful tool completion. A
//! server that needs to explain *why* a tool never produced a result —
//! or why a persisted outcome diverged from the in-memory result — has
//! no way to do so from `post_tool_use` alone.
//!
//! `ToolAuditSink` replaces that dependency. The turn loop now emits
//! an audit record at every lifecycle transition on **both** the inline
//! local-mode path and the externalised tool-runtime path.
//!
//! # Default sink
//!
//! The SDK ships [`NoopAuditSink`] for callers that don't need durable
//! audit. Servers should swap it for a sink that writes to their
//! audit-log backend.
//!
//! # Trait shape
//!
//! The record shape lives in [`agent_sdk_core::audit`] so it stays
//! data-only; the async trait lives here so `agent-sdk-core` does not
//! need to depend on `async-trait`.

use agent_sdk_core::audit::ToolAuditRecord;
use async_trait::async_trait;
use std::sync::Arc;

/// Async sink that receives one [`ToolAuditRecord`] per tool-call
/// lifecycle transition.
///
/// Sinks **must** be cheap: the turn loop awaits `record` on the hot
/// path. If the durable backend is slow, the implementation should
/// buffer and flush asynchronously rather than blocking the sink.
///
/// The sink must never panic; persistence failures should be logged
/// locally and a `persistence_failed` record fed back through the
/// normal path.
#[async_trait]
pub trait ToolAuditSink: Send + Sync + 'static {
    /// Record a single lifecycle event for a tool call.
    ///
    /// Called from the authoritative turn loop at every lifecycle
    /// transition. Implementations should be idempotent keyed on
    /// `(record.tool_call_id, record.outcome_kind())` — the same logical
    /// transition may arrive more than once if the loop retries after
    /// a transient failure.
    async fn record(&self, record: ToolAuditRecord);
}

/// Default sink that discards every record.
///
/// Useful as a placeholder before a server wires in its own audit
/// backend, and as a zero-overhead default for local/CLI usage.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopAuditSink;

#[async_trait]
impl ToolAuditSink for NoopAuditSink {
    async fn record(&self, _record: ToolAuditRecord) {
        // Intentionally no-op.
    }
}

/// Blanket impl so `Arc<S>` is itself a sink — lets callers share one
/// backend across clone boundaries without wrapping.
#[async_trait]
impl<S: ToolAuditSink + ?Sized> ToolAuditSink for Arc<S> {
    async fn record(&self, record: ToolAuditRecord) {
        (**self).record(record).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_core::audit::{AuditProvenance, ToolAuditOutcome, ToolAuditRecord};
    use agent_sdk_core::types::{ToolResult, ToolTier};
    use tokio::sync::Mutex;

    /// A test sink that captures every record it receives.
    ///
    /// Uses `tokio::sync::Mutex` so `.lock().await` is infallible and the
    /// sink never needs a panic path — per `CLAUDE.md` no `.unwrap()` is
    /// allowed even in tests.
    #[derive(Default)]
    pub struct VecSink {
        pub records: Mutex<Vec<ToolAuditRecord>>,
    }

    #[async_trait]
    impl ToolAuditSink for VecSink {
        async fn record(&self, record: ToolAuditRecord) {
            self.records.lock().await.push(record);
        }
    }

    fn sample_record(outcome: ToolAuditOutcome) -> ToolAuditRecord {
        ToolAuditRecord::new(agent_sdk_core::audit::ToolAuditRecordParams {
            tool_call_id: "call_x".into(),
            tool_name: "tool_x".into(),
            display_name: "Tool X".into(),
            tier: ToolTier::Observe,
            requested_input: serde_json::json!({}),
            effective_input: serde_json::json!({}),
            turn: 1,
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
            outcome,
        })
    }

    #[tokio::test]
    async fn noop_sink_accepts_all_variants_without_panicking() {
        let sink = NoopAuditSink;
        sink.record(sample_record(ToolAuditOutcome::Blocked {
            reason: "nope".into(),
        }))
        .await;
        sink.record(sample_record(ToolAuditOutcome::RequiresConfirmation {
            description: "ok?".into(),
            listen_context: None,
        }))
        .await;
        sink.record(sample_record(ToolAuditOutcome::Completed {
            result: ToolResult::success("ok"),
        }))
        .await;
    }

    #[tokio::test]
    async fn arc_wrapped_sink_forwards_records() {
        let inner = Arc::new(VecSink::default());
        let sink: Arc<dyn ToolAuditSink> = inner.clone();

        sink.record(sample_record(ToolAuditOutcome::Cached {
            result: ToolResult::success("cached"),
        }))
        .await;

        let kinds: Vec<&'static str> = inner
            .records
            .lock()
            .await
            .iter()
            .map(ToolAuditRecord::outcome_kind)
            .collect();
        assert_eq!(kinds, vec!["cached"]);
    }

    #[tokio::test]
    async fn sink_captures_every_lifecycle_variant() {
        let sink = Arc::new(VecSink::default());

        for outcome in [
            ToolAuditOutcome::Blocked { reason: "r".into() },
            ToolAuditOutcome::RequiresConfirmation {
                description: "d".into(),
                listen_context: None,
            },
            ToolAuditOutcome::Cached {
                result: ToolResult::success("c"),
            },
            ToolAuditOutcome::Replayed {
                result: ToolResult::success("r"),
            },
            ToolAuditOutcome::Invalidated { reason: "i".into() },
            ToolAuditOutcome::Completed {
                result: ToolResult::success("ok"),
            },
            ToolAuditOutcome::PersistenceFailed {
                result: None,
                error: "boom".into(),
            },
        ] {
            sink.record(sample_record(outcome)).await;
        }

        let kinds: Vec<&'static str> = sink
            .records
            .lock()
            .await
            .iter()
            .map(ToolAuditRecord::outcome_kind)
            .collect();
        assert_eq!(
            kinds,
            vec![
                "blocked",
                "requires_confirmation",
                "cached",
                "replayed",
                "invalidated",
                "completed",
                "persistence_failed",
            ],
        );
    }
}
