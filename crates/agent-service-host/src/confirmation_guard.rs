//! A host-supplied check on every `DecideConfirmation` before it applies.
//!
//! The gRPC control plane applies a confirmation decision for whoever sends
//! it. An embedding host that must keep some confirmations for a person (a
//! gate whose prompt *is* the approval) registers a
//! [`ConfirmationDecisionGuard`] with
//! [`crate::grpc::GrpcTransport::with_confirmation_decision_guard`]. The
//! request's `automatic` flag tells the guard that no person made the
//! decision; the guard sees the task and the call it awaits, and a refusal
//! leaves the task parked and answers `FAILED_PRECONDITION`.

use agent_sdk_foundation::{PendingToolCallInfo, ThreadId};
use agent_server::journal::AgentTask;
use agent_server::journal::prompt::PromptRecord;
use agent_server::worker::ConfirmationDecision;

/// One decision about to be applied.
#[derive(Debug)]
#[non_exhaustive]
pub struct ConfirmationDecisionRequest<'a> {
    pub thread_id: &'a ThreadId,
    /// The parked task the decision targets.
    pub task: &'a AgentTask,
    /// The call the task itself awaits: its own `spawn_index` slot of the
    /// batch, or the continuation's `awaiting_index` for a task that parked
    /// its own turn. `None` when the journal row names no call.
    pub awaited_call: Option<&'a PendingToolCallInfo>,
    pub decision: &'a ConfirmationDecision,
    /// The caller says no person made this decision. The caller sets it, so
    /// it stops honest automatic clients only; it is not an authorization
    /// boundary.
    pub automatic: bool,
}

impl<'a> ConfirmationDecisionRequest<'a> {
    /// A request for `task`, with [`Self::awaited_call`] read from the task.
    #[must_use]
    pub fn new(
        thread_id: &'a ThreadId,
        task: &'a AgentTask,
        decision: &'a ConfirmationDecision,
        automatic: bool,
    ) -> Self {
        Self {
            thread_id,
            task,
            awaited_call: awaited_call(task),
            decision,
            automatic,
        }
    }
}

/// Checks a confirmation decision before the control plane applies it.
pub trait ConfirmationDecisionGuard: Send + Sync {
    /// `Err(reason)` refuses the decision: nothing is applied and the RPC
    /// fails with `FAILED_PRECONDITION` carrying `reason`.
    ///
    /// # Errors
    ///
    /// Returns the refusal reason when the decision must not apply.
    fn check(&self, request: &ConfirmationDecisionRequest<'_>) -> Result<(), String>;

    /// The same check for a decision on an open prompt row
    /// ([`agent_server::journal::prompt`]). The default refuses nothing.
    ///
    /// # Errors
    ///
    /// Returns the refusal reason when the decision must not apply.
    fn check_prompt(&self, request: &PromptDecisionRequest<'_>) -> Result<(), String> {
        let _ = request;
        Ok(())
    }
}

/// One decision about to be applied to an open prompt row.
#[derive(Debug)]
#[non_exhaustive]
pub struct PromptDecisionRequest<'a> {
    /// The thread the prompt row is shown on.
    pub thread_id: &'a ThreadId,
    pub prompt: &'a PromptRecord,
    pub decision: &'a ConfirmationDecision,
    /// As [`ConfirmationDecisionRequest::automatic`].
    pub automatic: bool,
}

impl<'a> PromptDecisionRequest<'a> {
    #[must_use]
    pub const fn new(
        thread_id: &'a ThreadId,
        prompt: &'a PromptRecord,
        decision: &'a ConfirmationDecision,
        automatic: bool,
    ) -> Self {
        Self {
            thread_id,
            prompt,
            decision,
            automatic,
        }
    }
}

/// The call `task` itself awaits; see
/// [`ConfirmationDecisionRequest::awaited_call`].
#[must_use]
pub fn awaited_call(task: &AgentTask) -> Option<&PendingToolCallInfo> {
    let continuation = task.state.continuation()?;
    let index = match task.spawn_index {
        Some(index) => usize::try_from(index).ok()?,
        None => continuation.payload.awaiting_index,
    };
    continuation.payload.pending_tool_calls.get(index)
}
