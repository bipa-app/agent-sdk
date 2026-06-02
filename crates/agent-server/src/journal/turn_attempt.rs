//! Turn-attempt schema, identity, and append-only invariants.
//!
//! A [`TurnAttempt`] is an immutable audit record for a single LLM
//! request/response cycle within a task. The table is append-only
//! apart from closing the current open attempt — no row is ever
//! deleted or rewritten. This gives the server a durable execution
//! log with full model/provider provenance that survives retries,
//! failovers, and provider rotations.
//!
//! # Lifecycle
//!
//! Every attempt goes through exactly two states:
//!
//! 1. **Open** — created by [`TurnAttempt::open`] when the worker
//!    starts an LLM call. The row carries the request blob and
//!    provenance but has no response data yet.
//! 2. **Closed** — transitioned by [`TurnAttempt::close`] when the
//!    LLM call completes (successfully or not). The response blob,
//!    response model, stop reason, outcome, usage, and duration are
//!    all filled in at this point.
//!
//! Once closed, a row is permanently immutable.
//!
//! # What this table does **not** own
//!
//! - Continuation / pause state → lives on [`super::task_state::TaskState`]
//! - Scheduler state → lives on [`super::task::AgentTask`]
//! - Message projection → lives on [`super::message::MessageProjection`]
//! - Checkpoint storage → out of scope (future phase)

use agent_sdk_foundation::audit::AuditProvenance;
use serde::{Deserialize, Serialize};
use std::fmt;
use time::OffsetDateTime;
use uuid::Uuid;

use super::task::AgentTaskId;

// ─────────────────────────────────────────────────────────────────────
// Identity
// ─────────────────────────────────────────────────────────────────────

/// Unique identifier for a turn-attempt row.
///
/// Formatted as `attempt_<uuid>` to be visually distinct from task IDs
/// (`task_<uuid>`) and lease IDs (`lease_<uuid>`) in logs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TurnAttemptId(pub String);

impl TurnAttemptId {
    /// Allocate a fresh attempt ID.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("attempt_{}", Uuid::new_v4()))
    }

    /// Wrap an existing string as an attempt ID (used by stores when
    /// rehydrating rows from durable storage).
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TurnAttemptId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TurnAttemptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────

/// Terminal outcome for a single LLM request/response cycle.
///
/// Maps to [`agent_sdk_foundation::llm::ChatOutcome`] variants plus a
/// cancellation path. This is a server-audit enum — it records
/// *what happened*, not *what to do next*.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnAttemptOutcome {
    /// The LLM returned a successful response.
    Success,
    /// The provider returned a rate-limit signal.
    RateLimited,
    /// The request was rejected as invalid by the provider.
    InvalidRequest,
    /// The provider returned a server-side error.
    ServerError,
    /// The attempt was cancelled before the LLM responded.
    Cancelled,
}

impl TurnAttemptOutcome {
    /// Static discriminant string for metrics and audit rows.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::RateLimited => "rate_limited",
            Self::InvalidRequest => "invalid_request",
            Self::ServerError => "server_error",
            Self::Cancelled => "cancelled",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Structural errors for [`TurnAttempt`] rows.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TurnAttemptSchemaError {
    /// A close transition was attempted on an already-closed attempt.
    #[error("attempt is already closed")]
    AlreadyClosed,

    /// A closed attempt has no outcome set.
    #[error("closed attempt must have an outcome")]
    ClosedWithoutOutcome,

    /// An open attempt has an outcome set.
    #[error("open attempt must not have an outcome")]
    OutcomeOnOpenAttempt,

    /// An open attempt has response fields set.
    #[error("open attempt must not have response fields")]
    ResponseOnOpenAttempt,

    /// A closed attempt has no duration.
    #[error("closed attempt must have a duration")]
    ClosedWithoutDuration,

    /// An open attempt has a duration set.
    #[error("open attempt must not have a duration")]
    DurationOnOpenAttempt,
}

// ─────────────────────────────────────────────────────────────────────
// TurnAttempt
// ─────────────────────────────────────────────────────────────────────

/// One row in the `turn_attempts` audit table.
///
/// Captures the full request/response audit for a single LLM call
/// within a task. The row is opened when the worker starts the LLM
/// call and closed when the call completes (or is cancelled). Once
/// closed the row is permanently immutable.
///
/// # Fields
///
/// | Group | Fields |
/// |-------|--------|
/// | Identity | `id`, `task_id`, `attempt_number` |
/// | Request | `request_blob`, `requested_model`, `provider` |
/// | Response | `response_blob`, `response_id`, `response_model` |
/// | Outcome | `stop_reason`, `outcome` |
/// | Usage | `input_tokens`, `output_tokens`, `cached_input_tokens` |
/// | Timing | `opened_at`, `closed_at`, `duration_ms` |
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnAttempt {
    /// Unique row identity.
    pub id: TurnAttemptId,

    /// The task this attempt belongs to.
    pub task_id: AgentTaskId,

    /// 1-indexed attempt number within the task (first call = 1).
    pub attempt_number: u32,

    // ── Request provenance ───────────────────────────────────────
    /// Provider identifier (e.g. `"anthropic"`, `"openai"`).
    pub provider: String,

    /// Model the caller requested (may differ from what the
    /// provider actually used in the response).
    pub requested_model: String,

    /// Opaque serialised request payload. Stored as a JSON blob so
    /// the audit table does not need to understand the provider's
    /// wire format.
    pub request_blob: serde_json::Value,

    // ── Response provenance (filled on close) ────────────────────
    /// Opaque serialised response payload. `None` while the attempt
    /// is open.
    pub response_blob: Option<serde_json::Value>,

    /// Provider-assigned response identifier (e.g. Anthropic's
    /// `msg_...` ID). `None` while open or when the provider does
    /// not return one.
    pub response_id: Option<String>,

    /// Model string the provider actually used in the response.
    /// `None` while open; may differ from `requested_model` when
    /// the provider aliases or upgrades.
    pub response_model: Option<String>,

    /// Stop reason reported by the provider. `None` while open.
    pub stop_reason: Option<agent_sdk_foundation::llm::StopReason>,

    // ── Outcome ──────────────────────────────────────────────────
    /// Terminal outcome of the attempt. `None` while open.
    pub outcome: Option<TurnAttemptOutcome>,

    // ── Usage ────────────────────────────────────────────────────
    /// Input tokens consumed. `None` while open.
    pub input_tokens: Option<u32>,

    /// Output tokens produced. `None` while open.
    pub output_tokens: Option<u32>,

    /// Cached input tokens (provider-specific). `None` while open.
    pub cached_input_tokens: Option<u32>,

    // ── Timing ───────────────────────────────────────────────────
    /// When the attempt was opened (LLM call started).
    #[serde(with = "time::serde::rfc3339")]
    pub opened_at: OffsetDateTime,

    /// When the attempt was closed. `None` while open.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "time::serde::rfc3339::option"
    )]
    pub closed_at: Option<OffsetDateTime>,

    /// Wall-clock duration in milliseconds. `None` while open.
    pub duration_ms: Option<u64>,

    // ── OTel correlation ────────────────────────────────────────
    /// Hex-encoded `OTel` `TraceId` of the live span at attempt-open.
    ///
    /// Captured from `Context::current().span().span_context()` when
    /// the worker opens the attempt.  Null when the worker runs
    /// without an active `OTel` span (e.g. local dev with no exporter,
    /// or pre-A7 rows that predate this column).
    ///
    /// Used by `crate::worker::root_turn::call_llm_with_retry` on
    /// replay to attach an `agent.replay.original_*` `SpanLink` from
    /// the fresh attempt's span back to the original attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub otel_trace_id: Option<String>,

    /// Hex-encoded `OTel` `SpanId` of the live span at attempt-open.
    ///
    /// Companion to [`Self::otel_trace_id`]; same null semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub otel_span_id: Option<String>,
}

impl TurnAttempt {
    /// Open a new attempt for the given task.
    ///
    /// The returned row carries the request-side provenance and is
    /// ready to be persisted. Call [`Self::close`] when the LLM call
    /// completes.
    #[must_use]
    pub fn open(params: OpenAttemptParams) -> Self {
        Self {
            id: TurnAttemptId::new(),
            task_id: params.task_id,
            attempt_number: params.attempt_number,
            provider: params.provenance.provider,
            requested_model: params.provenance.model,
            request_blob: params.request_blob,
            response_blob: None,
            response_id: None,
            response_model: None,
            stop_reason: None,
            outcome: None,
            input_tokens: None,
            output_tokens: None,
            cached_input_tokens: None,
            opened_at: params.now,
            closed_at: None,
            duration_ms: None,
            otel_trace_id: params.otel_trace_id,
            otel_span_id: params.otel_span_id,
        }
    }

    /// Close this attempt with the LLM call's results.
    ///
    /// This is the only mutation a turn-attempt row ever undergoes.
    /// Once closed the row is permanently immutable.
    ///
    /// Duration is computed from `opened_at` to `now`. Negative
    /// durations (clock skew) are clamped to zero.
    ///
    /// # Errors
    ///
    /// Returns [`TurnAttemptSchemaError::AlreadyClosed`] if the
    /// attempt has already been closed.
    pub fn close(
        mut self,
        params: CloseAttemptParams,
        now: OffsetDateTime,
    ) -> Result<Self, TurnAttemptSchemaError> {
        if self.is_closed() {
            return Err(TurnAttemptSchemaError::AlreadyClosed);
        }

        self.response_blob = Some(params.response_blob);
        self.response_id = params.response_id;
        self.response_model = params.response_model;
        self.stop_reason = params.stop_reason;
        self.outcome = Some(params.outcome);
        self.input_tokens = Some(params.input_tokens);
        self.output_tokens = Some(params.output_tokens);
        self.cached_input_tokens = Some(params.cached_input_tokens);
        self.closed_at = Some(now);

        let dur = now - self.opened_at;
        // max(0) ensures we clamp negative durations (clock skew) to zero.
        // try_from is safe after clamping to [0, u64::MAX].
        let ms = dur.whole_milliseconds().max(0).min(i128::from(u64::MAX));
        self.duration_ms = Some(u64::try_from(ms).unwrap_or(u64::MAX));

        self.validate()?;
        Ok(self)
    }

    /// `true` when the attempt is still waiting for the LLM response.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.closed_at.is_none()
    }

    /// `true` when the attempt has been closed with a terminal outcome.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed_at.is_some()
    }

    /// Validate structural invariants.
    ///
    /// Called after every transition. An invalid row must never be
    /// persisted.
    ///
    /// # Invariants
    ///
    /// - **Open** rows must not have: `outcome`, `response_blob`,
    ///   `response_model`, `response_id`, `stop_reason`,
    ///   `input_tokens`, `output_tokens`, `cached_input_tokens`,
    ///   `duration_ms`, `closed_at`.
    /// - **Closed** rows must have: `outcome`, `duration_ms`,
    ///   `closed_at`.
    ///
    /// # Errors
    ///
    /// Returns a [`TurnAttemptSchemaError`] variant describing which
    /// invariant was violated.
    pub const fn validate(&self) -> Result<(), TurnAttemptSchemaError> {
        if self.is_closed() {
            // Closed invariants
            if self.outcome.is_none() {
                return Err(TurnAttemptSchemaError::ClosedWithoutOutcome);
            }
            if self.duration_ms.is_none() {
                return Err(TurnAttemptSchemaError::ClosedWithoutDuration);
            }
        } else {
            // Open invariants
            if self.outcome.is_some() {
                return Err(TurnAttemptSchemaError::OutcomeOnOpenAttempt);
            }
            if self.response_blob.is_some()
                || self.response_model.is_some()
                || self.response_id.is_some()
                || self.stop_reason.is_some()
                || self.input_tokens.is_some()
                || self.output_tokens.is_some()
                || self.cached_input_tokens.is_some()
            {
                return Err(TurnAttemptSchemaError::ResponseOnOpenAttempt);
            }
            if self.duration_ms.is_some() {
                return Err(TurnAttemptSchemaError::DurationOnOpenAttempt);
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Param structs
// ─────────────────────────────────────────────────────────────────────

/// Arguments for [`TurnAttempt::open`].
///
/// Named fields prevent positional confusion for a struct that lands
/// in the durable audit log.
#[derive(Clone, Debug)]
pub struct OpenAttemptParams {
    /// The task this attempt belongs to.
    pub task_id: AgentTaskId,
    /// 1-indexed attempt number within the task.
    pub attempt_number: u32,
    /// Provider / model provenance.
    pub provenance: AuditProvenance,
    /// Serialised request payload.
    pub request_blob: serde_json::Value,
    /// Current wall-clock time.
    pub now: OffsetDateTime,
    /// Optional hex-encoded `OTel` `TraceId` of the live span at
    /// attempt-open.  Persisted on the row and surfaced to the next
    /// attempt for replay-link emission.  Pass `None` when the worker
    /// is not running under an `OTel` context.
    pub otel_trace_id: Option<String>,
    /// Optional hex-encoded `OTel` `SpanId` of the live span at
    /// attempt-open; companion to [`Self::otel_trace_id`].
    pub otel_span_id: Option<String>,
}

/// Arguments for [`TurnAttempt::close`].
///
/// Response-side data filled in by the worker when the LLM call
/// completes.
#[derive(Clone, Debug)]
pub struct CloseAttemptParams {
    /// Serialised response payload.
    pub response_blob: serde_json::Value,
    /// Provider-assigned response identifier.
    pub response_id: Option<String>,
    /// Model string the provider actually used.
    pub response_model: Option<String>,
    /// Stop reason from the provider.
    pub stop_reason: Option<agent_sdk_foundation::llm::StopReason>,
    /// Terminal outcome of this attempt.
    pub outcome: TurnAttemptOutcome,
    /// Input tokens consumed.
    pub input_tokens: u32,
    /// Output tokens produced.
    pub output_tokens: u32,
    /// Cached input tokens.
    pub cached_input_tokens: u32,
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn sample_open_params() -> OpenAttemptParams {
        OpenAttemptParams {
            task_id: AgentTaskId::from_string("task_test-1"),
            attempt_number: 1,
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
            request_blob: serde_json::json!({
                "messages": [{"role": "user", "content": "hello"}],
                "model": "claude-sonnet-4-5-20250929"
            }),
            now: t0(),
            otel_trace_id: None,
            otel_span_id: None,
        }
    }

    fn sample_close_params() -> CloseAttemptParams {
        CloseAttemptParams {
            response_blob: serde_json::json!({
                "id": "msg_01",
                "content": [{"type": "text", "text": "hi"}]
            }),
            response_id: Some("msg_01".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 10,
        }
    }

    // ── Identity ─────────────────────────────────────────────────

    #[test]
    fn attempt_id_has_prefix() {
        let id = TurnAttemptId::new();
        assert!(
            id.as_str().starts_with("attempt_"),
            "expected attempt_ prefix, got: {id}",
        );
    }

    #[test]
    fn attempt_id_display_matches_inner() {
        let id = TurnAttemptId::from_string("attempt_abc");
        assert_eq!(format!("{id}"), "attempt_abc");
    }

    #[test]
    fn attempt_id_round_trips_through_json() -> Result<()> {
        let id = TurnAttemptId::from_string("attempt_rnd");
        let json = serde_json::to_string(&id).context("serialize")?;
        let back: TurnAttemptId = serde_json::from_str(&json).context("deserialize")?;
        assert_eq!(back, id);
        Ok(())
    }

    // ── Outcome ──────────────────────────────────────────────────

    #[test]
    fn outcome_as_str_matches_serde() -> Result<()> {
        for (outcome, expected) in [
            (TurnAttemptOutcome::Success, "success"),
            (TurnAttemptOutcome::RateLimited, "rate_limited"),
            (TurnAttemptOutcome::InvalidRequest, "invalid_request"),
            (TurnAttemptOutcome::ServerError, "server_error"),
            (TurnAttemptOutcome::Cancelled, "cancelled"),
        ] {
            assert_eq!(outcome.as_str(), expected);
            let json = serde_json::to_value(&outcome).context("serialize")?;
            assert_eq!(json.as_str().context("string")?, expected);
        }
        Ok(())
    }

    // ── Open ─────────────────────────────────────────────────────

    #[test]
    fn open_attempt_is_open_and_validates() {
        let attempt = TurnAttempt::open(sample_open_params());

        assert!(attempt.is_open());
        assert!(!attempt.is_closed());
        assert_eq!(attempt.attempt_number, 1);
        assert_eq!(attempt.provider, "anthropic");
        assert_eq!(attempt.requested_model, "claude-sonnet-4-5-20250929");
        assert!(attempt.response_blob.is_none());
        assert!(attempt.outcome.is_none());
        assert!(attempt.duration_ms.is_none());
        assert!(attempt.validate().is_ok());
    }

    // ── Close ────────────────────────────────────────────────────

    #[test]
    fn close_attempt_fills_response_fields() -> Result<()> {
        let attempt = TurnAttempt::open(sample_open_params());
        let closed = attempt
            .close(sample_close_params(), t_plus(5))
            .context("close")?;

        assert!(closed.is_closed());
        assert!(!closed.is_open());
        assert_eq!(closed.outcome, Some(TurnAttemptOutcome::Success));
        assert_eq!(closed.response_id, Some("msg_01".into()));
        assert_eq!(
            closed.response_model,
            Some("claude-sonnet-4-5-20250929".into()),
        );
        assert_eq!(
            closed.stop_reason,
            Some(agent_sdk_foundation::llm::StopReason::EndTurn),
        );
        assert_eq!(closed.input_tokens, Some(100));
        assert_eq!(closed.output_tokens, Some(50));
        assert_eq!(closed.cached_input_tokens, Some(10));
        assert_eq!(closed.duration_ms, Some(5_000));
        closed.validate().context("validate")?;
        Ok(())
    }

    #[test]
    fn close_on_already_closed_returns_error() -> Result<()> {
        let attempt = TurnAttempt::open(sample_open_params());
        let closed = attempt
            .close(sample_close_params(), t_plus(1))
            .context("close")?;
        let err = closed
            .close(sample_close_params(), t_plus(2))
            .expect_err("should reject double close");
        assert_eq!(err, TurnAttemptSchemaError::AlreadyClosed);
        Ok(())
    }

    #[test]
    fn close_with_clock_skew_clamps_to_zero() -> Result<()> {
        let params = OpenAttemptParams {
            now: t_plus(10),
            ..sample_open_params()
        };
        let attempt = TurnAttempt::open(params);
        // Close *before* open time → duration should clamp to 0
        let closed = attempt
            .close(sample_close_params(), t0())
            .context("close")?;
        assert_eq!(closed.duration_ms, Some(0));
        Ok(())
    }

    // ── Validation edge cases ────────────────────────────────────

    #[test]
    fn open_with_outcome_fails_validation() {
        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.outcome = Some(TurnAttemptOutcome::Success);
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::OutcomeOnOpenAttempt),
        );
    }

    #[test]
    fn open_with_response_blob_fails_validation() {
        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.response_blob = Some(serde_json::json!({}));
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::ResponseOnOpenAttempt),
        );
    }

    #[test]
    fn open_with_response_id_fails_validation() {
        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.response_id = Some("msg_01".into());
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::ResponseOnOpenAttempt),
        );
    }

    #[test]
    fn open_with_stop_reason_fails_validation() {
        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.stop_reason = Some(agent_sdk_foundation::llm::StopReason::EndTurn);
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::ResponseOnOpenAttempt),
        );
    }

    #[test]
    fn open_with_usage_tokens_fails_validation() {
        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.input_tokens = Some(100);
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::ResponseOnOpenAttempt),
        );

        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.output_tokens = Some(50);
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::ResponseOnOpenAttempt),
        );

        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.cached_input_tokens = Some(10);
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::ResponseOnOpenAttempt),
        );
    }

    #[test]
    fn open_with_duration_fails_validation() {
        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.duration_ms = Some(123);
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::DurationOnOpenAttempt),
        );
    }

    #[test]
    fn closed_without_outcome_fails_validation() {
        let mut attempt = TurnAttempt::open(sample_open_params());
        // Force-close without outcome
        attempt.closed_at = Some(t_plus(1));
        attempt.duration_ms = Some(1_000);
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::ClosedWithoutOutcome),
        );
    }

    #[test]
    fn closed_without_duration_fails_validation() {
        let mut attempt = TurnAttempt::open(sample_open_params());
        attempt.closed_at = Some(t_plus(1));
        attempt.outcome = Some(TurnAttemptOutcome::Success);
        assert_eq!(
            attempt.validate(),
            Err(TurnAttemptSchemaError::ClosedWithoutDuration),
        );
    }

    // ── JSON round-trip ──────────────────────────────────────────

    #[test]
    fn open_attempt_round_trips_through_json() -> Result<()> {
        let attempt = TurnAttempt::open(sample_open_params());
        let json = serde_json::to_string(&attempt).context("serialize")?;
        let back: TurnAttempt = serde_json::from_str(&json).context("deserialize")?;

        assert_eq!(back.id, attempt.id);
        assert_eq!(back.task_id, attempt.task_id);
        assert_eq!(back.attempt_number, attempt.attempt_number);
        assert_eq!(back.provider, attempt.provider);
        assert_eq!(back.requested_model, attempt.requested_model);
        assert!(back.is_open());
        back.validate().context("validate")?;
        Ok(())
    }

    #[test]
    fn closed_attempt_round_trips_through_json() -> Result<()> {
        let attempt = TurnAttempt::open(sample_open_params());
        let closed = attempt
            .close(sample_close_params(), t_plus(3))
            .context("close")?;
        let json = serde_json::to_string(&closed).context("serialize")?;
        let back: TurnAttempt = serde_json::from_str(&json).context("deserialize")?;

        assert_eq!(back.id, closed.id);
        assert!(back.is_closed());
        assert_eq!(back.outcome, closed.outcome);
        assert_eq!(back.duration_ms, closed.duration_ms);
        assert_eq!(back.response_id, closed.response_id);
        back.validate().context("validate")?;
        Ok(())
    }

    #[test]
    fn closed_attempt_json_contains_expected_fields() -> Result<()> {
        let attempt = TurnAttempt::open(sample_open_params());
        let closed = attempt
            .close(sample_close_params(), t_plus(2))
            .context("close")?;
        let json = serde_json::to_value(&closed).context("serialize")?;

        // Verify provenance fields are present
        assert_eq!(json["provider"], "anthropic");
        assert_eq!(json["requested_model"], "claude-sonnet-4-5-20250929");
        assert_eq!(json["response_model"], "claude-sonnet-4-5-20250929");
        assert_eq!(json["response_id"], "msg_01");

        // Verify outcome
        assert_eq!(json["outcome"], "success");
        assert_eq!(json["stop_reason"], "end_turn");

        // Verify usage
        assert_eq!(json["input_tokens"], 100);
        assert_eq!(json["output_tokens"], 50);
        assert_eq!(json["cached_input_tokens"], 10);

        // Verify timing
        assert_eq!(json["duration_ms"], 2_000);
        Ok(())
    }

    // ── Multiple attempts ────────────────────────────────────────

    #[test]
    fn second_attempt_has_distinct_id() {
        let a1 = TurnAttempt::open(sample_open_params());
        let a2 = TurnAttempt::open(OpenAttemptParams {
            attempt_number: 2,
            ..sample_open_params()
        });
        assert_ne!(a1.id, a2.id);
        assert_eq!(a2.attempt_number, 2);
    }

    // ── Non-success outcomes ─────────────────────────────────────

    #[test]
    fn rate_limited_attempt_closes_with_empty_response() -> Result<()> {
        let attempt = TurnAttempt::open(sample_open_params());
        let closed = attempt
            .close(
                CloseAttemptParams {
                    response_blob: serde_json::json!(null),
                    response_id: None,
                    response_model: None,
                    stop_reason: None,
                    outcome: TurnAttemptOutcome::RateLimited,
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                },
                t_plus(1),
            )
            .context("close")?;
        assert_eq!(closed.outcome, Some(TurnAttemptOutcome::RateLimited));
        assert!(closed.response_id.is_none());
        closed.validate().context("validate")?;
        Ok(())
    }

    #[test]
    fn cancelled_attempt_closes_successfully() -> Result<()> {
        let attempt = TurnAttempt::open(sample_open_params());
        let closed = attempt
            .close(
                CloseAttemptParams {
                    response_blob: serde_json::json!(null),
                    response_id: None,
                    response_model: None,
                    stop_reason: None,
                    outcome: TurnAttemptOutcome::Cancelled,
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                },
                t_plus(1),
            )
            .context("close")?;
        assert_eq!(closed.outcome, Some(TurnAttemptOutcome::Cancelled));
        closed.validate().context("validate")?;
        Ok(())
    }

    // ── OTel correlation ─────────────────────────────────────────

    #[test]
    fn open_attempt_defaults_otel_ids_to_none() {
        let attempt = TurnAttempt::open(sample_open_params());
        assert!(attempt.otel_trace_id.is_none());
        assert!(attempt.otel_span_id.is_none());
    }

    #[test]
    fn open_attempt_carries_otel_ids_when_supplied() -> Result<()> {
        let params = OpenAttemptParams {
            otel_trace_id: Some("4bf92f3577b34da6a3ce929d0e0e4736".to_string()),
            otel_span_id: Some("00f067aa0ba902b7".to_string()),
            ..sample_open_params()
        };
        let attempt = TurnAttempt::open(params);
        assert_eq!(
            attempt.otel_trace_id.as_deref(),
            Some("4bf92f3577b34da6a3ce929d0e0e4736"),
        );
        assert_eq!(attempt.otel_span_id.as_deref(), Some("00f067aa0ba902b7"));
        attempt.validate().context("validate")?;
        Ok(())
    }

    #[test]
    fn otel_ids_round_trip_through_json() -> Result<()> {
        let params = OpenAttemptParams {
            otel_trace_id: Some("4bf92f3577b34da6a3ce929d0e0e4736".to_string()),
            otel_span_id: Some("00f067aa0ba902b7".to_string()),
            ..sample_open_params()
        };
        let attempt = TurnAttempt::open(params);
        let json = serde_json::to_string(&attempt).context("serialize")?;
        let back: TurnAttempt = serde_json::from_str(&json).context("deserialize")?;
        assert_eq!(back.otel_trace_id, attempt.otel_trace_id);
        assert_eq!(back.otel_span_id, attempt.otel_span_id);
        Ok(())
    }

    #[test]
    fn missing_otel_ids_round_trip_through_json_for_legacy_rows() -> Result<()> {
        // Older attempts (pre-A7) have no otel_* keys in their JSON.
        // Make sure deserialization tolerates the absence and surfaces
        // None.
        let legacy = serde_json::json!({
            "id": "attempt_legacy",
            "task_id": "task_legacy",
            "attempt_number": 1,
            "provider": "anthropic",
            "requested_model": "claude-sonnet-4-5-20250929",
            "request_blob": {"messages": []},
            "opened_at": "2024-11-15T00:00:00Z",
        });
        let attempt: TurnAttempt =
            serde_json::from_value(legacy).context("deserialize legacy row")?;
        assert!(attempt.otel_trace_id.is_none());
        assert!(attempt.otel_span_id.is_none());
        attempt.validate().context("validate")?;
        Ok(())
    }
}
