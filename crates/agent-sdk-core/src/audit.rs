//! Authoritative tool audit records.
//!
//! The audit surface that the server uses to explain **every** tool
//! lifecycle outcome — not just successful completion. This replaces the
//! `post_tool_use` hook as the sole audit surface on the authoritative
//! (server) execution path.
//!
//! # Why this exists
//!
//! `post_tool_use` only fires once per tool call and only describes the
//! terminal [`ToolResult`]. The server has to explain paths that never
//! reach a successful result, including:
//!
//! - **Blocked** — the policy hook rejected the tool.
//! - **`RequiresConfirmation`** — the policy hook yielded for user approval.
//! - **Cached** — an earlier completed execution was replayed from the
//!   execution store.
//! - **Replayed** — the caller resubmitted external tool results for an
//!   already-processed handoff.
//! - **Invalidated** — a listen-tool snapshot expired or was invalidated
//!   before the user could confirm.
//! - **Completed** — the tool ran to completion (success or failure).
//! - **`PersistenceFailed`** — the tool ran but the event / execution
//!   store refused to durably record the outcome.
//!
//! These outcomes are modelled as [`ToolAuditOutcome`] variants on a
//! single [`ToolAuditRecord`]. Sinks receive one record per lifecycle
//! transition and can persist them to a durable audit table without
//! having to reconstruct the path from scattered hook calls.
//!
//! # Trait location
//!
//! Only the **record shape** lives in `agent-sdk-core` (this module is
//! data-only). The async [`ToolAuditSink`](../../agent_sdk_tools/audit/trait.ToolAuditSink.html)
//! trait lives in `agent-sdk-tools` so `agent-sdk-core` stays free of
//! async-trait dependencies.

use crate::types::{ListenExecutionContext, ToolResult, ToolTier};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// Provider / model provenance for an audit record.
///
/// Captured at the moment the record is emitted so that durable audit
/// rows survive provider/model rotations. Present on every record
/// because every tool-call lifecycle event happens in the context of
/// the LLM turn that requested the tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditProvenance {
    /// Provider identifier (e.g. `"anthropic"`, `"openai"`, `"vertex"`).
    pub provider: String,
    /// Model identifier (e.g. `"claude-sonnet-4-5-20250929"`).
    pub model: String,
}

impl AuditProvenance {
    /// Construct a provenance record from borrowed strings.
    #[must_use]
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }
}

/// Lifecycle outcome for a single tool call.
///
/// Every variant is an **authoritative** terminal state the server must
/// persist — including paths that bypass tool execution entirely (blocked,
/// confirmation, cached replay) or that fail persistence after the tool
/// already ran.
///
/// Variants are ordered roughly by lifecycle position: policy check → cache
/// lookup → execution → post-execution persistence.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolAuditOutcome {
    /// The policy hook rejected the tool call.
    ///
    /// The tool never executed. The reason is the string returned by
    /// [`ToolDecision::Block`](../../agent_sdk_tools/hooks/enum.ToolDecision.html#variant.Block).
    Blocked {
        /// Reason provided by the policy hook.
        reason: String,
    },

    /// The policy hook yielded for user approval.
    ///
    /// The tool is paused pending a resume decision. The turn loop will
    /// emit a follow-up record on resume (either [`Completed`](Self::Completed)
    /// after execution or [`Blocked`](Self::Blocked) if policy now rejects).
    RequiresConfirmation {
        /// Human-readable confirmation description shown to the user.
        description: String,
        /// Optional listen-context captured at confirmation time.
        listen_context: Option<ListenExecutionContext>,
    },

    /// The execution store already held a completed result for this
    /// tool call — the idempotency layer replayed the cached outcome
    /// instead of calling the tool again.
    Cached {
        /// The cached [`ToolResult`] that was replayed.
        result: ToolResult,
    },

    /// The caller resubmitted external tool results for an already
    /// processed handoff, and the SDK served the previously recorded
    /// result rather than re-accepting the payload.
    ///
    /// Distinct from [`Cached`](Self::Cached) in that this fires on the
    /// **external** runtime path where the SDK did not execute the tool
    /// itself in any attempt.
    Replayed {
        /// The [`ToolResult`] previously recorded for this tool call.
        result: ToolResult,
    },

    /// A listen-tool snapshot expired or was invalidated before the
    /// user could confirm it.
    ///
    /// This is a non-completion path: no final [`ToolResult`] is
    /// produced because the confirmation window closed.
    Invalidated {
        /// Reason the listen-tool invalidated its snapshot.
        reason: String,
    },

    /// The tool ran to completion (success or failure).
    ///
    /// `result.success` indicates whether the tool itself succeeded;
    /// even a failing run is considered a completed lifecycle.
    Completed {
        /// Final [`ToolResult`] produced by the tool.
        result: ToolResult,
    },

    /// The tool executed but the server could not durably persist the
    /// outcome (event store, execution store, or message append failed).
    ///
    /// The record preserves the in-memory [`ToolResult`] so that audit
    /// consumers can reason about divergence between what the tool
    /// produced and what made it to durable storage.
    PersistenceFailed {
        /// The [`ToolResult`] that would have been persisted, if any.
        ///
        /// `None` when the persistence layer failed before a result was
        /// produced (e.g. a `tool_call_start` event failed to append).
        result: Option<ToolResult>,
        /// Short, human-readable description of the persistence failure.
        error: String,
    },
}

impl ToolAuditOutcome {
    /// Static discriminant string used for metrics, tracing attributes,
    /// and durable audit rows.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Blocked { .. } => "blocked",
            Self::RequiresConfirmation { .. } => "requires_confirmation",
            Self::Cached { .. } => "cached",
            Self::Replayed { .. } => "replayed",
            Self::Invalidated { .. } => "invalidated",
            Self::Completed { .. } => "completed",
            Self::PersistenceFailed { .. } => "persistence_failed",
        }
    }

    /// Returns the [`ToolResult`] associated with this outcome, if one
    /// is available.
    ///
    /// Present for [`Cached`](Self::Cached), [`Replayed`](Self::Replayed),
    /// [`Completed`](Self::Completed), and most
    /// [`PersistenceFailed`](Self::PersistenceFailed) paths. Absent for
    /// [`Blocked`](Self::Blocked), [`RequiresConfirmation`](Self::RequiresConfirmation),
    /// and [`Invalidated`](Self::Invalidated).
    #[must_use]
    pub const fn result(&self) -> Option<&ToolResult> {
        match self {
            Self::Cached { result } | Self::Replayed { result } | Self::Completed { result } => {
                Some(result)
            }
            Self::PersistenceFailed { result, .. } => result.as_ref(),
            Self::Blocked { .. } | Self::RequiresConfirmation { .. } | Self::Invalidated { .. } => {
                None
            }
        }
    }
}

/// Single authoritative audit record for one tool-call lifecycle event.
///
/// A tool call may produce **multiple** records over its lifetime — for
/// example a `RequiresConfirmation` followed by a `Completed` after the
/// user approves, or a `Completed` followed by a `PersistenceFailed` if
/// the event store rejects the terminal event.
///
/// Records are self-describing: consumers do **not** need to correlate
/// them with hook calls or event-store rows to understand what happened.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolAuditRecord {
    /// Unique tool call ID (from the LLM's `tool_use`).
    pub tool_call_id: String,
    /// Wire-format tool name.
    pub tool_name: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Permission tier of the tool at the moment the record was emitted.
    pub tier: ToolTier,
    /// Input as requested by the LLM (audit trail).
    pub requested_input: serde_json::Value,
    /// Effective input after SDK preparation (may differ for listen-tools).
    pub effective_input: serde_json::Value,
    /// Turn number this record belongs to.
    pub turn: usize,
    /// Provider / model provenance for this turn's LLM call.
    pub provenance: AuditProvenance,
    /// Lifecycle outcome carrying the variant-specific payload.
    pub outcome: ToolAuditOutcome,
    /// UTC timestamp when the record was produced.
    #[serde(with = "time::serde::rfc3339")]
    pub recorded_at: OffsetDateTime,
}

impl ToolAuditRecord {
    /// Build a record using the current wall-clock time.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        display_name: impl Into<String>,
        tier: ToolTier,
        requested_input: serde_json::Value,
        effective_input: serde_json::Value,
        turn: usize,
        provenance: AuditProvenance,
        outcome: ToolAuditOutcome,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            display_name: display_name.into(),
            tier,
            requested_input,
            effective_input,
            turn,
            provenance,
            outcome,
            recorded_at: OffsetDateTime::now_utc(),
        }
    }

    /// Return the outcome's discriminant string.
    #[must_use]
    pub const fn outcome_kind(&self) -> &'static str {
        self.outcome.kind()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(outcome: ToolAuditOutcome) -> ToolAuditRecord {
        ToolAuditRecord::new(
            "call_1",
            "read_file",
            "Read File",
            ToolTier::Observe,
            serde_json::json!({"path": "/tmp/x"}),
            serde_json::json!({"path": "/tmp/x"}),
            2,
            AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
            outcome,
        )
    }

    #[test]
    fn outcome_kind_matches_variant() {
        assert_eq!(
            ToolAuditOutcome::Blocked {
                reason: "no".into(),
            }
            .kind(),
            "blocked",
        );
        assert_eq!(
            ToolAuditOutcome::RequiresConfirmation {
                description: "pls".into(),
                listen_context: None,
            }
            .kind(),
            "requires_confirmation",
        );
        assert_eq!(
            ToolAuditOutcome::Cached {
                result: ToolResult::success("ok"),
            }
            .kind(),
            "cached",
        );
        assert_eq!(
            ToolAuditOutcome::Replayed {
                result: ToolResult::success("ok"),
            }
            .kind(),
            "replayed",
        );
        assert_eq!(
            ToolAuditOutcome::Invalidated {
                reason: "expired".into(),
            }
            .kind(),
            "invalidated",
        );
        assert_eq!(
            ToolAuditOutcome::Completed {
                result: ToolResult::success("ok"),
            }
            .kind(),
            "completed",
        );
        assert_eq!(
            ToolAuditOutcome::PersistenceFailed {
                result: None,
                error: "boom".into(),
            }
            .kind(),
            "persistence_failed",
        );
    }

    #[test]
    fn outcome_result_accessor() {
        let ok = ToolResult::success("ok");
        assert!(
            ToolAuditOutcome::Blocked { reason: "n".into() }
                .result()
                .is_none()
        );
        assert_eq!(
            ToolAuditOutcome::Completed { result: ok.clone() }
                .result()
                .map(|r| r.output.as_str()),
            Some("ok"),
        );
        assert_eq!(
            ToolAuditOutcome::PersistenceFailed {
                result: Some(ok),
                error: "e".into(),
            }
            .result()
            .map(|r| r.output.as_str()),
            Some("ok"),
        );
    }

    #[test]
    fn record_round_trips_through_json() {
        let record = sample_record(ToolAuditOutcome::Completed {
            result: ToolResult::success("hello"),
        });
        let json = serde_json::to_string(&record).unwrap();
        let back: ToolAuditRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_call_id, "call_1");
        assert_eq!(back.outcome_kind(), "completed");
        assert_eq!(back.provenance.provider, "anthropic");
        assert_eq!(back.provenance.model, "claude-sonnet-4-5-20250929");
    }

    #[test]
    fn every_outcome_serialises_with_snake_case_tag() {
        // Non-trivial assertion: the external tag format must be stable
        // for durable audit tables and dashboards.
        let record = sample_record(ToolAuditOutcome::Blocked {
            reason: "policy".into(),
        });
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["outcome"]["kind"], "blocked");
        assert_eq!(json["outcome"]["reason"], "policy");
    }
}
