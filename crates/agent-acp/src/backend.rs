//! The backend seam: what the ACP transport needs from an agent runtime.
//!
//! [`AcpBackend`] is the reusable boundary between the wire (this crate) and
//! an embedding application's runtime. The design rule (contract C-c/C-d in
//! the satoshi plans, review finding #9): **backends supply identity and
//! I/O, never recovery logic.** Cursor arithmetic, duplicate suppression,
//! terminal mapping — all of it lives once in the SDK-owned run loop
//! ([`crate::run`]), so every backend implementation inherits the same
//! correctness behavior.
//!
//! The loop maps text and thinking content, tool lifecycle, the subagent
//! lifecycle, plan synthesis, usage, keepalives, terminal events, and
//! confirmation parks (contract C-e): the backend names the parked child
//! behind a tool call and applies a decision; correlation, retry, and
//! answer routing live in the loop.

use std::sync::Arc;

use agent_sdk_foundation::AgentEvent;
use tokio_util::sync::CancellationToken;

use crate::server::{PromptError, PromptHandler, PromptRequest, UpdateSink};
use crate::session::NewSessionParams;
use crate::wire::StopReason;

/// Identity of one submitted turn, returned by [`AcpBackend::submit_prompt`].
///
/// `first_event_sequence` is the sequence number the turn's **first** event
/// will receive, captured before admission (the durable host's
/// `stream_after` semantics). The run loop — and only the run loop —
/// converts it into the exclusive lower bound the event stream expects.
#[derive(Debug, Clone)]
pub struct AcpRunHandle {
    /// Backend thread the turn runs on (e.g. `buzz:<channel-uuid>`).
    pub thread_id: String,
    /// Backend task id for the admitted root turn.
    pub task_id: String,
    /// Sequence number the turn's first committed event will carry.
    pub first_event_sequence: u64,
}

/// One committed event with its thread-scoped sequence number.
#[derive(Debug, Clone)]
pub struct RunEvent {
    /// Monotonic, thread-scoped sequence assigned at durable commit.
    pub sequence: u64,
    /// The committed event payload.
    pub event: AgentEvent,
}

/// Items yielded by a backend event stream.
#[derive(Debug)]
pub enum RunStreamItem {
    /// A committed event. Boxed to keep the enum small next to the
    /// data-free `Lagged` variant (same shape as the durable host's own
    /// `StreamEvent::Event(Box<CommittedEvent>)`).
    Event(Box<RunEvent>),
    /// The stream lost contiguity and can no longer guarantee gap-free
    /// delivery. The run loop — never the backend — recovers by reopening
    /// from its last yielded sequence (contract C-c).
    Lagged,
    /// The journal has PRUNED events in the range this stream needed:
    /// unlike [`RunStreamItem::Lagged`], no reopen can restore
    /// continuity. The run loop fails the prompt loudly. Backends that
    /// cannot distinguish retention loss from transient lag should
    /// prefer `Lagged` only when a reopen can genuinely recover.
    RetentionGap,
}

/// Boxed stream of committed events from a backend.
pub type EventStream = futures::stream::BoxStream<'static, RunStreamItem>;

/// Failure surfaced by a backend operation.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct BackendError {
    /// Human-readable failure description.
    pub message: String,
}

impl BackendError {
    /// Build a backend error from any displayable message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// A submitted task's durable status, as the backend's store sees it
/// (contract C-d's reconciliation source of truth).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendTaskStatus {
    /// Anything non-terminal: pending, queued, running, parked on a
    /// confirmation — the turn may still produce events.
    Running,
    /// The task completed; the turn is over even if its terminal event
    /// never reached this stream.
    Completed,
    /// The task was cancelled.
    Cancelled,
    /// The task failed, with the store's recorded error when there is
    /// one. A task can reach this status WITHOUT a journal `Error`
    /// event (crash between the status write and the commit) — which is
    /// exactly why the run loop polls status on stall.
    Failed {
        /// The error the store recorded for the failure, if any.
        error: Option<String>,
    },
}

/// A tool-runtime child parked on a confirmation, paired with the tool
/// call it will execute when approved (contract C-e / V3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwaitingConfirmation {
    /// The parked child task — what a decision targets.
    pub child_task_id: String,
    /// The tool call the child runs; matches `ToolRequiresConfirmation.id`.
    pub tool_call_id: String,
}

/// The client's answer to a confirmation, as applied to the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    /// Run the tool. The backend's policy is re-checked at resume
    /// regardless — approval never bypasses it.
    Allow,
    /// Do not run the tool; the turn continues with the rejection.
    Reject,
}

/// What applying a decision did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecideOutcome {
    /// The park was decided.
    Decided,
    /// No such park is pending any more — already decided, cancelled, or
    /// swept. A stale answer is a successful no-op, never an error.
    NonePending,
}

/// An agent runtime the ACP transport can drive.
///
/// Object-safe: `Arc<dyn AcpBackend>` works. Thread identity is the
/// backend's decision — per contract C-b, `session/new` binds nothing;
/// the backend resolves/creates its durable thread at first prompt from
/// whatever identity material the session and prompt carry.
#[async_trait::async_trait]
pub trait AcpBackend: Send + Sync + 'static {
    /// Submit one turn and return its identity.
    ///
    /// # Errors
    ///
    /// A [`BackendError`] resolves the pending `session/prompt` request as a
    /// JSON-RPC internal error.
    async fn submit_prompt(
        &self,
        session_id: &str,
        session: &NewSessionParams,
        blocks: &[String],
    ) -> Result<AcpRunHandle, BackendError>;

    /// Open the committed-event stream for a thread, yielding events with
    /// `sequence > after_sequence` (`None` = from the beginning).
    ///
    /// # Errors
    ///
    /// A [`BackendError`] resolves the pending prompt as a JSON-RPC
    /// internal error.
    async fn open_events(
        &self,
        thread_id: &str,
        after_sequence: Option<u64>,
    ) -> Result<EventStream, BackendError>;

    /// Request cancellation of an in-flight turn. The turn is expected to
    /// close with a terminal `Cancelled` event on the stream; this call only
    /// signals.
    ///
    /// # Errors
    ///
    /// A [`BackendError`] here is logged, not fatal — the run loop keeps
    /// draining the stream either way.
    async fn cancel(&self, thread_id: &str, task_id: &str) -> Result<(), BackendError>;

    /// Report the durable status of a submitted task (contract C-d).
    ///
    /// The run loop calls this to reconcile: when the stream stalls, and
    /// when an UNATTRIBUTED terminal event arrives mid-stream. It is the
    /// only authority allowed to close a turn without an attributed
    /// terminal event.
    ///
    /// # Errors
    ///
    /// A [`BackendError`] here is logged and retried on the next stall
    /// tick, never fatal on its own.
    async fn task_status(
        &self,
        thread_id: &str,
        task_id: &str,
    ) -> Result<BackendTaskStatus, BackendError>;

    /// Read the result text of a completed subagent invocation.
    ///
    /// Design §3.1 rule 4: result text never rides the event stream —
    /// `SubagentResult` goes to the invocation task's result and the
    /// parent's history only. The run loop calls this once per subagent
    /// close so the terminal `tool_call_update` can carry the child's
    /// final response (or its recorded error) instead of a bare
    /// success/failure. `Ok(None)` means the backend has nothing to
    /// offer; the loop then closes the row with the progress summary.
    ///
    /// # Errors
    ///
    /// A [`BackendError`] here is logged, never fatal — the row still
    /// closes, with the summary as content.
    async fn subagent_result(
        &self,
        thread_id: &str,
        subagent_task_id: &str,
    ) -> Result<Option<String>, BackendError>;

    /// The children currently parked on a confirmation for this thread,
    /// oldest first, each with the tool call it will run (contract C-e).
    ///
    /// The run loop calls this when a `ToolRequiresConfirmation` event
    /// arrives, to name the child in the permission request's option ids.
    /// The park may commit a beat after the event; the loop retries
    /// briefly, so a momentarily incomplete list is not an error.
    ///
    /// # Errors
    ///
    /// A [`BackendError`] is logged and retried within the correlation
    /// window, never fatal on its own.
    async fn awaiting_confirmations(
        &self,
        thread_id: &str,
    ) -> Result<Vec<AwaitingConfirmation>, BackendError>;

    /// Apply the client's decision to ONE parked child, by id.
    ///
    /// Targets exactly `child_task_id`: with parallel parks, answers arrive
    /// in any order and an oldest-first decide would eat a sibling's
    /// confirmation.
    ///
    /// # Errors
    ///
    /// A [`BackendError`] is logged; the park stays for the turn's cancel
    /// or the backend's orphan policy to resolve.
    async fn decide_confirmation(
        &self,
        thread_id: &str,
        child_task_id: &str,
        decision: PermissionDecision,
    ) -> Result<DecideOutcome, BackendError>;
}

/// Bridges an [`AcpBackend`] into the wire server's [`PromptHandler`] seam.
pub struct BackendPromptHandler<B: ?Sized> {
    backend: Arc<B>,
}

impl<B: AcpBackend + ?Sized> BackendPromptHandler<B> {
    /// Wrap a backend.
    pub const fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }
}

#[async_trait::async_trait]
impl<B: AcpBackend + ?Sized> PromptHandler for BackendPromptHandler<B> {
    async fn prompt(
        &self,
        request: PromptRequest,
        updates: UpdateSink,
        cancel: CancellationToken,
    ) -> Result<StopReason, PromptError> {
        let handle = self
            .backend
            .submit_prompt(&request.session_id, &request.session, &request.blocks)
            .await
            .map_err(|e| PromptError::new(e.message))?;
        crate::run::run_prompt(self.backend.as_ref(), &handle, &updates, &cancel).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The trait must stay object-safe: satoshi wires `Arc<dyn AcpBackend>`
    // through composition layers in M0.3.
    const _OBJECT_SAFE: fn(&dyn AcpBackend) = |_| {};

    #[test]
    fn backend_error_displays_message() {
        assert_eq!(BackendError::new("boom").to_string(), "boom");
    }
}
