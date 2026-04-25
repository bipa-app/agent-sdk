//! Durable tool audit events for child-task execution lifecycle.
//!
//! This module provides the server-side audit surface for tool-call
//! lifecycle transitions within the child-task model. Unlike the SDK's
//! [`ToolAuditRecord`](agent_sdk_core::audit::ToolAuditRecord) (designed
//! for the inline execution path), this audit surface tracks the
//! **durable multi-step lifecycle** that tool calls go through in the
//! server's child-task model:
//!
//! ```text
//!   Dispatched ──▶ ConfirmationRequested ──▶ ConfirmationApproved ──▶ ExecutionStarted ──▶ Completed
//!                        │                        │                                          │
//!                        ├──▶ ConfirmationRejected │                                     Failed
//!                        └──▶ ConfirmationTimedOut  └──▶ PolicyDenied
//!                                                                                       Cancelled
//!                                                                                       FailClosed
//! ```
//!
//! Every transition produces a [`ToolAuditEvent`] record with full
//! provenance (provider, model, task IDs, operation ID, effect class)
//! so the audit trail can explain what happened without depending on
//! provider logs or transient worker memory.
//!
//! # Redaction
//!
//! Tool inputs and outputs may contain sensitive data. The
//! [`RedactionPolicy`] module provides
//! baseline redaction rules that should be applied before persisting
//! audit events to durable storage.

use agent_sdk_core::ThreadId;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use time::OffsetDateTime;
use uuid::Uuid;

use super::execution_intent::ToolEffectClass;
use super::redaction::{RedactionPolicy, redact_error, redact_string, redact_value};
use super::task::AgentTaskId;

// ─────────────────────────────────────────────────────────────────────
// Identity
// ─────────────────────────────────────────────────────────────────────

/// Unique identifier for a tool audit event row.
///
/// Formatted as `tae_<uuid>` to be visually distinct from task IDs
/// (`task_<uuid>`) and attempt IDs (`attempt_<uuid>`) in logs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ToolAuditEventId(pub String);

impl ToolAuditEventId {
    #[must_use]
    pub fn new() -> Self {
        Self(format!("tae_{}", Uuid::new_v4()))
    }

    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ToolAuditEventId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ToolAuditEventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Lifecycle event kind
// ─────────────────────────────────────────────────────────────────────

/// Lifecycle event kind for a tool-call audit record.
///
/// Each variant represents a single transition in the tool-call
/// lifecycle. A tool call may produce multiple events as it moves
/// through the lifecycle (e.g. `Dispatched` → `ConfirmationRequested`
/// → `ConfirmationApproved` → `ExecutionStarted` → `Completed`).
///
/// Variants are ordered by lifecycle position and use `snake_case`
/// serde tags for stable durable storage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolAuditEventKind {
    /// Child task dispatched (spawned from parent root turn).
    Dispatched,

    /// Tool paused for user confirmation (Confirm-tier tool).
    ConfirmationRequested,

    /// User approved the confirmation prompt.
    ConfirmationApproved,

    /// User rejected the confirmation prompt.
    ConfirmationRejected { reason: String },

    /// Confirmation timed out without a user decision.
    ConfirmationTimedOut,

    /// Authoritative policy denied execution despite user approval.
    PolicyDenied { reason: String },

    /// Execution intent persisted; executor callback starting.
    ExecutionStarted,

    /// Tool completed successfully.
    Completed,

    /// Tool execution failed (executor returned an error).
    Failed { error: String },

    /// Tool execution cancelled before the executor ran.
    Cancelled,

    /// Fail-closed: execution blocked due to safety concern
    /// (intent persistence failure, ambiguous in-flight state, or
    /// duplicate execution detected).
    FailClosed { reason: String },
}

impl ToolAuditEventKind {
    /// Static discriminant string for metrics and audit queries.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Dispatched => "dispatched",
            Self::ConfirmationRequested => "confirmation_requested",
            Self::ConfirmationApproved => "confirmation_approved",
            Self::ConfirmationRejected { .. } => "confirmation_rejected",
            Self::ConfirmationTimedOut => "confirmation_timed_out",
            Self::PolicyDenied { .. } => "policy_denied",
            Self::ExecutionStarted => "execution_started",
            Self::Completed => "completed",
            Self::Failed { .. } => "failed",
            Self::Cancelled => "cancelled",
            Self::FailClosed { .. } => "fail_closed",
        }
    }

    /// Whether this event kind represents a terminal lifecycle state.
    ///
    /// Terminal events end the tool-call lifecycle. No further audit
    /// events should be emitted for the same operation after a terminal
    /// event.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::Failed { .. }
                | Self::Cancelled
                | Self::FailClosed { .. }
                | Self::ConfirmationRejected { .. }
                | Self::ConfirmationTimedOut
                | Self::PolicyDenied { .. }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// Audit event record
// ─────────────────────────────────────────────────────────────────────

/// Single durable audit event for a tool-call lifecycle transition.
///
/// Each tool call may produce multiple events as it moves through the
/// lifecycle. Events are self-describing: consumers do not need to
/// correlate them with journal task rows or execution intents to
/// understand what happened.
///
/// # Provenance
///
/// Every event carries the provider/model context from the LLM turn
/// that requested the tool call, so audit rows survive provider
/// rotations and model upgrades.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolAuditEvent {
    /// Unique event identity.
    pub id: ToolAuditEventId,

    /// Stable operation identity (`child_task_id:tool_call_id`).
    pub operation_id: String,

    /// Child task that owns this tool execution.
    pub task_id: AgentTaskId,

    /// Parent root-turn task that spawned this child.
    pub parent_task_id: AgentTaskId,

    /// Thread the task family is bound to.
    pub thread_id: ThreadId,

    /// Raw LLM-assigned tool call id.
    pub tool_call_id: String,

    /// Tool name.
    pub tool_name: String,

    /// Side-effect classification that drove the guard decision.
    pub effect_class: ToolEffectClass,

    /// Lifecycle event kind with variant-specific payload.
    pub kind: ToolAuditEventKind,

    // ── Provenance ──────────────────────────────────────────────
    /// Provider identifier (e.g. `"anthropic"`, `"openai"`).
    pub provider: String,

    /// Model identifier (e.g. `"claude-sonnet-4-5-20250929"`).
    pub model: String,

    // ── Redactable fields ───────────────────────────────────────
    /// Tool input snapshot (redacted according to policy before storage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,

    /// Tool output (redacted according to policy before storage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,

    /// Error detail (redacted according to policy before storage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// When the event was recorded.
    #[serde(with = "time::serde::rfc3339")]
    pub recorded_at: OffsetDateTime,
}

/// Parameters for building a [`ToolAuditEvent`].
///
/// Named fields prevent positional confusion for a struct that lands
/// in the durable audit log.
#[derive(Clone, Debug)]
pub struct ToolAuditEventParams {
    pub operation_id: String,
    pub task_id: AgentTaskId,
    pub parent_task_id: AgentTaskId,
    pub thread_id: ThreadId,
    pub tool_call_id: String,
    pub tool_name: String,
    pub effect_class: ToolEffectClass,
    pub kind: ToolAuditEventKind,
    pub provider: String,
    pub model: String,
    pub input: Option<serde_json::Value>,
    pub output: Option<String>,
    pub error: Option<String>,
    pub now: OffsetDateTime,
}

impl ToolAuditEvent {
    /// Build an audit event from named parameters.
    #[must_use]
    pub fn new(params: ToolAuditEventParams) -> Self {
        Self {
            id: ToolAuditEventId::new(),
            operation_id: params.operation_id,
            task_id: params.task_id,
            parent_task_id: params.parent_task_id,
            thread_id: params.thread_id,
            tool_call_id: params.tool_call_id,
            tool_name: params.tool_name,
            effect_class: params.effect_class,
            kind: params.kind,
            provider: params.provider,
            model: params.model,
            input: params.input,
            output: params.output,
            error: params.error,
            recorded_at: params.now,
        }
    }

    /// Return the event kind's discriminant string.
    #[must_use]
    pub const fn kind_str(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Whether this event represents a terminal lifecycle state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.kind.is_terminal()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Store trait
// ─────────────────────────────────────────────────────────────────────

/// Durable store for tool audit events.
///
/// Every tool-call lifecycle transition is recorded through this trait.
/// Implementations should ensure events are durably written and
/// queryable by operation, task, and thread.
#[async_trait]
pub trait ToolAuditEventStore: Send + Sync {
    /// Record a single lifecycle event.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be durably written.
    async fn record_event(&self, event: &ToolAuditEvent) -> anyhow::Result<()>;

    /// List all events for a given operation (tool call lifecycle),
    /// ordered by durable write order.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be read.
    async fn list_by_operation(&self, operation_id: &str) -> anyhow::Result<Vec<ToolAuditEvent>>;

    /// List all events for a given child task, ordered by durable write
    /// order.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be read.
    async fn list_by_task(&self, task_id: &AgentTaskId) -> anyhow::Result<Vec<ToolAuditEvent>>;

    /// List all events for a given thread, ordered by durable write
    /// order.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be read.
    async fn list_by_thread(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<ToolAuditEvent>>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory store
// ─────────────────────────────────────────────────────────────────────

/// In-memory implementation of [`ToolAuditEventStore`].
///
/// Suitable for testing and single-process deployments. Indexes by
/// `operation_id`, `task_id`, and `thread_id` for efficient queries.
///
/// All data is held behind a single [`RwLock`] so that writes to
/// `events` and the three index maps are atomic with respect to
/// concurrent readers.
pub struct InMemoryToolAuditEventStore {
    inner: RwLock<InMemoryToolAuditInner>,
}

#[derive(Default)]
struct InMemoryToolAuditInner {
    events: Vec<ToolAuditEvent>,
    by_operation: HashMap<String, Vec<usize>>,
    by_task: HashMap<String, Vec<usize>>,
    by_thread: HashMap<String, Vec<usize>>,
}

impl Default for InMemoryToolAuditEventStore {
    fn default() -> Self {
        Self {
            inner: RwLock::new(InMemoryToolAuditInner::default()),
        }
    }
}

impl InMemoryToolAuditEventStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return all recorded events (for test assertions).
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned.
    pub fn all_events(&self) -> anyhow::Result<Vec<ToolAuditEvent>> {
        use anyhow::Context;
        Ok(self
            .inner
            .read()
            .ok()
            .context("lock poisoned")?
            .events
            .clone())
    }
}

#[async_trait]
impl ToolAuditEventStore for InMemoryToolAuditEventStore {
    async fn record_event(&self, event: &ToolAuditEvent) -> anyhow::Result<()> {
        use anyhow::Context;

        let mut inner = self.inner.write().ok().context("lock poisoned")?;

        let idx = inner.events.len();
        inner.events.push(event.clone());

        inner
            .by_operation
            .entry(event.operation_id.clone())
            .or_default()
            .push(idx);

        inner
            .by_task
            .entry(event.task_id.0.clone())
            .or_default()
            .push(idx);

        inner
            .by_thread
            .entry(event.thread_id.0.clone())
            .or_default()
            .push(idx);

        drop(inner);
        Ok(())
    }

    async fn list_by_operation(&self, operation_id: &str) -> anyhow::Result<Vec<ToolAuditEvent>> {
        use anyhow::Context;

        let inner = self.inner.read().ok().context("lock poisoned")?;
        let Some(indices) = inner.by_operation.get(operation_id) else {
            return Ok(Vec::new());
        };
        let mut result: Vec<ToolAuditEvent> = indices
            .iter()
            .filter_map(|&i| inner.events.get(i).cloned())
            .collect();
        drop(inner);
        result.sort_by_key(|e| e.recorded_at);
        Ok(result)
    }

    async fn list_by_task(&self, task_id: &AgentTaskId) -> anyhow::Result<Vec<ToolAuditEvent>> {
        use anyhow::Context;

        let inner = self.inner.read().ok().context("lock poisoned")?;
        let Some(indices) = inner.by_task.get(&task_id.0) else {
            return Ok(Vec::new());
        };
        let mut result: Vec<ToolAuditEvent> = indices
            .iter()
            .filter_map(|&i| inner.events.get(i).cloned())
            .collect();
        drop(inner);
        result.sort_by_key(|e| e.recorded_at);
        Ok(result)
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<ToolAuditEvent>> {
        use anyhow::Context;

        let inner = self.inner.read().ok().context("lock poisoned")?;
        let Some(indices) = inner.by_thread.get(&thread_id.0) else {
            return Ok(Vec::new());
        };
        let mut result: Vec<ToolAuditEvent> = indices
            .iter()
            .filter_map(|&i| inner.events.get(i).cloned())
            .collect();
        drop(inner);
        result.sort_by_key(|e| e.recorded_at);
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Redaction decorator
// ─────────────────────────────────────────────────────────────────────

/// Returns a copy of `event` with its redactable fields run through
/// the provided [`RedactionPolicy`].
///
/// Redacted fields:
/// - [`ToolAuditEvent::input`] via [`redact_value`]
/// - [`ToolAuditEvent::output`] via [`redact_string`]
/// - [`ToolAuditEvent::error`] via [`redact_error`]
/// - The `error` string carried by the
///   [`ToolAuditEventKind::Failed`] variant, also via [`redact_error`],
///   since failed tool executions can surface raw provider error text.
///
/// Non-redactable fields (identity, provenance, lifecycle kind) are
/// copied unchanged so the audit trail still answers "what happened and
/// in what order" without leaking secrets.
#[must_use]
pub fn redact_event(event: &ToolAuditEvent, policy: &RedactionPolicy) -> ToolAuditEvent {
    let kind = match &event.kind {
        ToolAuditEventKind::Failed { error } => ToolAuditEventKind::Failed {
            error: redact_error(error, policy),
        },
        other => other.clone(),
    };

    ToolAuditEvent {
        id: event.id.clone(),
        operation_id: event.operation_id.clone(),
        task_id: event.task_id.clone(),
        parent_task_id: event.parent_task_id.clone(),
        thread_id: event.thread_id.clone(),
        tool_call_id: event.tool_call_id.clone(),
        tool_name: event.tool_name.clone(),
        effect_class: event.effect_class,
        kind,
        provider: event.provider.clone(),
        model: event.model.clone(),
        input: event.input.as_ref().map(|v| redact_value(v, policy)),
        output: event.output.as_deref().map(|s| redact_string(s, policy)),
        error: event.error.as_deref().map(|s| redact_error(s, policy)),
        recorded_at: event.recorded_at,
    }
}

#[must_use]
fn durable_store_redaction_policy() -> RedactionPolicy {
    RedactionPolicy::baseline()
}

/// Decorator that applies a [`RedactionPolicy`] to every event on its
/// way into durable storage, and passes reads through unchanged.
///
/// Writes are the only path that must redact: once rows are persisted,
/// reading them back does not re-introduce the un-redacted payload.
pub struct RedactingToolAuditEventStore {
    inner: Arc<dyn ToolAuditEventStore>,
    policy: RedactionPolicy,
}

impl RedactingToolAuditEventStore {
    /// Wrap an inner store with the supplied redaction policy.
    #[must_use]
    pub fn new(inner: Arc<dyn ToolAuditEventStore>, policy: RedactionPolicy) -> Self {
        Self { inner, policy }
    }

    /// Wrap an inner store using the default durable-write redaction
    /// policy: baseline redaction for input, output, and error fields.
    #[must_use]
    pub fn baseline(inner: Arc<dyn ToolAuditEventStore>) -> Self {
        Self::new(inner, durable_store_redaction_policy())
    }
}

#[async_trait]
impl ToolAuditEventStore for RedactingToolAuditEventStore {
    async fn record_event(&self, event: &ToolAuditEvent) -> anyhow::Result<()> {
        let redacted = redact_event(event, &self.policy);
        self.inner.record_event(&redacted).await
    }

    async fn list_by_operation(&self, operation_id: &str) -> anyhow::Result<Vec<ToolAuditEvent>> {
        self.inner.list_by_operation(operation_id).await
    }

    async fn list_by_task(&self, task_id: &AgentTaskId) -> anyhow::Result<Vec<ToolAuditEvent>> {
        self.inner.list_by_task(task_id).await
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> anyhow::Result<Vec<ToolAuditEvent>> {
        self.inner.list_by_thread(thread_id).await
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::task::AgentTaskId;
    use agent_sdk_core::ThreadId;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn sample_params(kind: ToolAuditEventKind, now: OffsetDateTime) -> ToolAuditEventParams {
        ToolAuditEventParams {
            operation_id: "task_child1:call_1".into(),
            task_id: AgentTaskId::from_string("task_child1"),
            parent_task_id: AgentTaskId::from_string("task_parent1"),
            thread_id: ThreadId::from_string("thread_test"),
            tool_call_id: "call_1".into(),
            tool_name: "transfer".into(),
            effect_class: ToolEffectClass::SideEffecting,
            kind,
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5-20250929".into(),
            input: Some(serde_json::json!({"amount": 100})),
            output: None,
            error: None,
            now,
        }
    }

    // ── Identity ────────────────────────────────────────────────

    #[test]
    fn event_id_has_prefix() {
        let id = ToolAuditEventId::new();
        assert!(
            id.as_str().starts_with("tae_"),
            "expected tae_ prefix, got: {id}",
        );
    }

    #[test]
    fn event_id_round_trips_through_json() -> anyhow::Result<()> {
        let id = ToolAuditEventId::from_string("tae_abc123");
        let json = serde_json::to_string(&id)?;
        let back: ToolAuditEventId = serde_json::from_str(&json)?;
        assert_eq!(back, id);
        Ok(())
    }

    // ── Event kind ──────────────────────────────────────────────

    #[test]
    fn event_kind_as_str_matches_serde() {
        let cases = vec![
            (ToolAuditEventKind::Dispatched, "dispatched"),
            (
                ToolAuditEventKind::ConfirmationRequested,
                "confirmation_requested",
            ),
            (
                ToolAuditEventKind::ConfirmationApproved,
                "confirmation_approved",
            ),
            (
                ToolAuditEventKind::ConfirmationRejected {
                    reason: "no".into(),
                },
                "confirmation_rejected",
            ),
            (
                ToolAuditEventKind::ConfirmationTimedOut,
                "confirmation_timed_out",
            ),
            (
                ToolAuditEventKind::PolicyDenied {
                    reason: "denied".into(),
                },
                "policy_denied",
            ),
            (ToolAuditEventKind::ExecutionStarted, "execution_started"),
            (ToolAuditEventKind::Completed, "completed"),
            (
                ToolAuditEventKind::Failed {
                    error: "err".into(),
                },
                "failed",
            ),
            (ToolAuditEventKind::Cancelled, "cancelled"),
            (
                ToolAuditEventKind::FailClosed {
                    reason: "unsafe".into(),
                },
                "fail_closed",
            ),
        ];

        for (kind, expected) in cases {
            assert_eq!(kind.as_str(), expected);
        }
    }

    #[test]
    fn terminal_event_kinds() {
        assert!(!ToolAuditEventKind::Dispatched.is_terminal());
        assert!(!ToolAuditEventKind::ConfirmationRequested.is_terminal());
        assert!(!ToolAuditEventKind::ConfirmationApproved.is_terminal());
        assert!(!ToolAuditEventKind::ExecutionStarted.is_terminal());

        assert!(ToolAuditEventKind::Completed.is_terminal());
        assert!(ToolAuditEventKind::Failed { error: "e".into() }.is_terminal());
        assert!(ToolAuditEventKind::Cancelled.is_terminal());
        assert!(ToolAuditEventKind::FailClosed { reason: "r".into() }.is_terminal());
        assert!(ToolAuditEventKind::ConfirmationRejected { reason: "r".into() }.is_terminal());
        assert!(ToolAuditEventKind::ConfirmationTimedOut.is_terminal());
        assert!(ToolAuditEventKind::PolicyDenied { reason: "r".into() }.is_terminal());
    }

    // ── Event construction ──────────────────────────────────────

    #[test]
    fn event_construction_and_accessors() {
        let event = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Dispatched, t0()));

        assert_eq!(event.kind_str(), "dispatched");
        assert!(!event.is_terminal());
        assert_eq!(event.operation_id, "task_child1:call_1");
        assert_eq!(event.tool_name, "transfer");
        assert_eq!(event.provider, "anthropic");
        assert_eq!(event.model, "claude-sonnet-4-5-20250929");
        assert_eq!(event.effect_class, ToolEffectClass::SideEffecting);
    }

    #[test]
    fn event_round_trips_through_json() -> anyhow::Result<()> {
        let event = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Completed, t_plus(10)));
        let json = serde_json::to_string(&event)?;
        let back: ToolAuditEvent = serde_json::from_str(&json)?;

        assert_eq!(back.id, event.id);
        assert_eq!(back.operation_id, event.operation_id);
        assert_eq!(back.tool_name, event.tool_name);
        assert_eq!(back.kind_str(), "completed");
        assert_eq!(back.provider, "anthropic");
        Ok(())
    }

    #[test]
    fn event_json_contains_expected_fields() -> anyhow::Result<()> {
        let event = ToolAuditEvent::new(sample_params(
            ToolAuditEventKind::Failed {
                error: "timeout".into(),
            },
            t0(),
        ));
        let json = serde_json::to_value(&event)?;

        assert_eq!(json["tool_name"], "transfer");
        assert_eq!(json["provider"], "anthropic");
        assert_eq!(json["model"], "claude-sonnet-4-5-20250929");
        assert_eq!(json["effect_class"], "side_effecting");
        assert_eq!(json["kind"]["kind"], "failed");
        assert_eq!(json["kind"]["error"], "timeout");

        Ok(())
    }

    // ── In-memory store ─────────────────────────────────────────

    #[tokio::test]
    async fn store_records_and_queries_by_operation() -> anyhow::Result<()> {
        let store = InMemoryToolAuditEventStore::new();

        let e1 = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Dispatched, t0()));
        let e2 = ToolAuditEvent::new(sample_params(
            ToolAuditEventKind::ExecutionStarted,
            t_plus(1),
        ));
        let e3 = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Completed, t_plus(2)));

        store.record_event(&e1).await?;
        store.record_event(&e2).await?;
        store.record_event(&e3).await?;

        let events = store.list_by_operation("task_child1:call_1").await?;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].kind_str(), "dispatched");
        assert_eq!(events[1].kind_str(), "execution_started");
        assert_eq!(events[2].kind_str(), "completed");

        Ok(())
    }

    #[tokio::test]
    async fn store_queries_by_task() -> anyhow::Result<()> {
        let store = InMemoryToolAuditEventStore::new();

        let e1 = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Dispatched, t0()));
        store.record_event(&e1).await?;

        // Different task
        let mut params2 = sample_params(ToolAuditEventKind::Dispatched, t_plus(1));
        params2.task_id = AgentTaskId::from_string("task_child2");
        params2.operation_id = "task_child2:call_2".into();
        let e2 = ToolAuditEvent::new(params2);
        store.record_event(&e2).await?;

        let task1_events = store
            .list_by_task(&AgentTaskId::from_string("task_child1"))
            .await?;
        assert_eq!(task1_events.len(), 1);
        assert_eq!(task1_events[0].task_id.0, "task_child1");

        let task2_events = store
            .list_by_task(&AgentTaskId::from_string("task_child2"))
            .await?;
        assert_eq!(task2_events.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn store_queries_by_thread() -> anyhow::Result<()> {
        let store = InMemoryToolAuditEventStore::new();

        let e1 = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Dispatched, t0()));
        store.record_event(&e1).await?;

        let events = store
            .list_by_thread(&ThreadId::from_string("thread_test"))
            .await?;
        assert_eq!(events.len(), 1);

        let empty = store
            .list_by_thread(&ThreadId::from_string("thread_other"))
            .await?;
        assert!(empty.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn store_returns_empty_for_unknown_operation() -> anyhow::Result<()> {
        let store = InMemoryToolAuditEventStore::new();
        let events = store.list_by_operation("nonexistent").await?;
        assert!(events.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn all_events_helper() -> anyhow::Result<()> {
        let store = InMemoryToolAuditEventStore::new();

        let e1 = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Dispatched, t0()));
        let e2 = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Completed, t_plus(1)));
        store.record_event(&e1).await?;
        store.record_event(&e2).await?;

        let all = store.all_events()?;
        assert_eq!(all.len(), 2);
        Ok(())
    }

    // ── Full lifecycle trace ────────────────────────────────────

    #[tokio::test]
    async fn full_lifecycle_produces_ordered_events() -> anyhow::Result<()> {
        let store = InMemoryToolAuditEventStore::new();
        let op_id = "task_child1:call_1";

        let lifecycle = vec![
            (ToolAuditEventKind::Dispatched, t0()),
            (ToolAuditEventKind::ConfirmationRequested, t_plus(1)),
            (ToolAuditEventKind::ConfirmationApproved, t_plus(5)),
            (ToolAuditEventKind::ExecutionStarted, t_plus(6)),
            (ToolAuditEventKind::Completed, t_plus(10)),
        ];

        for (kind, ts) in lifecycle {
            let event = ToolAuditEvent::new(sample_params(kind, ts));
            store.record_event(&event).await?;
        }

        let events = store.list_by_operation(op_id).await?;
        assert_eq!(events.len(), 5);

        let kinds: Vec<&str> = events.iter().map(ToolAuditEvent::kind_str).collect();
        assert_eq!(
            kinds,
            vec![
                "dispatched",
                "confirmation_requested",
                "confirmation_approved",
                "execution_started",
                "completed",
            ],
        );

        // Verify ordering by timestamp
        for window in events.windows(2) {
            assert!(window[0].recorded_at <= window[1].recorded_at);
        }

        Ok(())
    }

    // ── Redaction decorator ─────────────────────────────────────

    const REDACTED_MARKER_VALUE: &str = "[REDACTED]";

    fn sample_params_sensitive(
        kind: ToolAuditEventKind,
        now: OffsetDateTime,
    ) -> ToolAuditEventParams {
        let mut params = sample_params(kind, now);
        params.input = Some(serde_json::json!({
            "command": "curl -H 'Authorization: Bearer sk-secret' https://x",
            "api_key": "sk-abc123",
            "normal": "hello",
        }));
        params.output = Some("sk-leaked-token".into());
        params.error = Some("Bearer eyJ-token".into());
        params
    }

    #[test]
    fn redact_event_redacts_input_and_output_under_baseline_policy() {
        let policy = RedactionPolicy::baseline();
        let event = ToolAuditEvent::new(sample_params_sensitive(
            ToolAuditEventKind::Failed {
                error: "sk-failure-token".into(),
            },
            t0(),
        ));
        let redacted = redact_event(&event, &policy);

        let input = redacted.input.as_ref().expect("input present");
        assert_eq!(input["api_key"], REDACTED_MARKER_VALUE);
        assert_eq!(input["normal"], "hello");

        assert_eq!(redacted.output.as_deref(), Some(REDACTED_MARKER_VALUE));
        // Baseline error_level is now also Baseline — the top-level
        // error field and the error inside the Failed variant both
        // get wholesale-redacted because they start with sensitive
        // prefixes ("Bearer ", "sk-").
        assert_eq!(redacted.error.as_deref(), Some(REDACTED_MARKER_VALUE));

        match redacted.kind {
            ToolAuditEventKind::Failed { error } => {
                assert_eq!(error, REDACTED_MARKER_VALUE);
            }
            other => panic!("expected Failed variant, got {other:?}"),
        }

        // Identity and provenance survive redaction untouched.
        assert_eq!(redacted.id, event.id);
        assert_eq!(redacted.operation_id, event.operation_id);
        assert_eq!(redacted.provider, "anthropic");
    }

    #[test]
    fn redact_event_with_full_policy_masks_failed_error() {
        let policy = RedactionPolicy::full();
        let event = ToolAuditEvent::new(sample_params_sensitive(
            ToolAuditEventKind::Failed {
                error: "sk-failure-token".into(),
            },
            t0(),
        ));
        let redacted = redact_event(&event, &policy);

        match redacted.kind {
            ToolAuditEventKind::Failed { error } => assert_eq!(error, REDACTED_MARKER_VALUE),
            other => panic!("expected Failed variant, got {other:?}"),
        }
        assert_eq!(
            redacted.input.as_ref().expect("input present"),
            &serde_json::json!(REDACTED_MARKER_VALUE),
        );
        assert_eq!(redacted.output.as_deref(), Some(REDACTED_MARKER_VALUE));
        assert_eq!(redacted.error.as_deref(), Some(REDACTED_MARKER_VALUE));
    }

    #[tokio::test]
    async fn redacting_store_applies_policy_before_inner_write() -> anyhow::Result<()> {
        let inner = Arc::new(InMemoryToolAuditEventStore::new());
        let store = RedactingToolAuditEventStore::baseline(inner.clone());

        let event = ToolAuditEvent::new(sample_params_sensitive(
            ToolAuditEventKind::Failed {
                error: "sk-failure-token".into(),
            },
            t0(),
        ));
        store.record_event(&event).await?;

        let stored = inner.all_events()?;
        assert_eq!(stored.len(), 1);
        let stored = &stored[0];
        assert_eq!(stored.output.as_deref(), Some(REDACTED_MARKER_VALUE));
        assert_eq!(
            stored.input.as_ref().expect("input present")["api_key"],
            REDACTED_MARKER_VALUE,
        );
        assert_eq!(stored.error.as_deref(), Some(REDACTED_MARKER_VALUE));
        match &stored.kind {
            ToolAuditEventKind::Failed { error } => {
                assert_eq!(error, REDACTED_MARKER_VALUE);
            }
            other => panic!("expected Failed variant, got {other:?}"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn redacting_store_read_paths_pass_through() -> anyhow::Result<()> {
        let inner = Arc::new(InMemoryToolAuditEventStore::new());
        let store = RedactingToolAuditEventStore::baseline(inner.clone());
        let event = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Dispatched, t0()));
        store.record_event(&event).await?;

        let op_events = store.list_by_operation(&event.operation_id).await?;
        assert_eq!(op_events.len(), 1);
        assert_eq!(op_events[0].id, event.id);

        let task_events = store.list_by_task(&event.task_id).await?;
        assert_eq!(task_events.len(), 1);

        let thread_events = store.list_by_thread(&event.thread_id).await?;
        assert_eq!(thread_events.len(), 1);

        Ok(())
    }

    // ── Provenance fields ───────────────────────────────────────

    #[test]
    fn provenance_fields_round_trip() -> anyhow::Result<()> {
        let event = ToolAuditEvent::new(sample_params(ToolAuditEventKind::Completed, t0()));
        let json = serde_json::to_value(&event)?;

        assert_eq!(json["provider"], "anthropic");
        assert_eq!(json["model"], "claude-sonnet-4-5-20250929");
        assert_eq!(json["effect_class"], "side_effecting");
        assert_eq!(json["tool_call_id"], "call_1");
        assert_eq!(json["tool_name"], "transfer");
        assert!(json["thread_id"].is_string());
        assert!(json["task_id"].is_string());
        assert!(json["parent_task_id"].is_string());
        assert!(json["operation_id"].is_string());

        Ok(())
    }
}
