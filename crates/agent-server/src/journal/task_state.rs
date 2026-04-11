//! Typed durable task state for [`super::AgentTask`] rows.
//!
//! Phase 2.1's `AgentTask::state` field was deliberately a loose
//! `serde_json::Value` so the schema could land before the rewrite knew
//! exactly what each task kind needed to remember across pause / resume.
//! Phase 2.4 (ENG-7918) replaces it with [`TaskState`], a strongly typed
//! enum keyed on the row's pause status.
//!
//! The variants line up 1:1 with the two **paused** [`super::TaskStatus`]
//! values that exist in the rewrite today:
//!
//! | `TaskStatus` | `TaskState` variant | Why we need durable state |
//! |--------------|---------------------|---------------------------|
//! | [`super::TaskStatus::WaitingOnChildren`] | [`TaskState::WaitingOnChildren`] | A root turn that has spawned tool-runtime children carries a versioned [`ContinuationEnvelope`] so the parent loop can resume from the same point once every child reaches a terminal state. |
//! | [`super::TaskStatus::AwaitingConfirmation`] | [`TaskState::AwaitingConfirmation`] | A task paused on a user confirmation needs both the continuation **and** any prepared listen/execute operation that the SDK staged before the pause, so the resume path can either execute or cancel it. |
//! | `Pending` / `Running` after child aggregation | [`TaskState::ReadyToResume`] | A parent whose child batch has drained carries the saved continuation and suspended messages until a worker resumes the turn. |
//! | every other status | [`TaskState::None`] | Fresh runnable rows (`Pending`, `Running`, `Queued`) and terminal rows (`Completed`, `Failed`, `Cancelled`) carry no durable resume payload — the row's status itself is enough. |
//!
//! Cross-row invariants (`state` ↔ `status`) are enforced by
//! [`super::AgentTask::validate`] so the schema layer catches
//! illegal combinations the same way it catches lease-field drift or
//! terminal-without-`completed_at`. Tests in [`super::task`] cover the
//! ↔ classification table directly so any future variant addition
//! forces an explicit invariant decision rather than silently slipping
//! through `validate()`.
//!
//! ## Why `serde_json::Value` is not enough
//!
//! Two reasons drove the move to a typed enum:
//!
//! 1. **Resume safety.** A `serde_json::Value` would let a buggy or
//!    out-of-date worker rehydrate a paused row with the wrong shape
//!    (e.g. a confirmation continuation under a `WaitingOnChildren`
//!    row, or a missing `ContinuationEnvelope` entirely). Phase 2.4's
//!    acceptance criterion — "Confirmation-paused tasks persist the
//!    state needed for later resume" — relies on the durable row
//!    *itself* refusing to round-trip in an inconsistent state.
//! 2. **Pause-status agreement.** Cleanly typing the pause payload lets
//!    [`super::AgentTask::validate`] reject any row whose `status` and
//!    `state` disagree, which is the strongest we can do without a
//!    full state-machine in the schema layer. The Phase 2.6 child
//!    orchestration and Phase 2.5 retry workers will both inspect
//!    `state` to decide what to do on resume; a typed enum means they
//!    can `match` on it instead of probing JSON keys.
//!
//! ## Wire format
//!
//! [`TaskState`] uses `#[serde(tag = "kind", rename_all = "snake_case")]`
//! so durable rows look like:
//!
//! ```json
//! { "kind": "none" }
//! { "kind": "waiting_on_children", "continuation": { "version": 1, "payload": { ... } } }
//! { "kind": "awaiting_confirmation", "continuation": { "version": 1, "payload": { ... } }, "prepared_operation": null }
//! ```
//!
//! The variant names are stable across releases so durable rows persisted
//! by older binaries continue to deserialize cleanly. New variants must
//! be added with `#[serde(default)]` defaults on any new fields they
//! introduce, the same forward-compatibility rule
//! [`agent_sdk_core::ContinuationEnvelope`] follows.

use agent_sdk_core::{ContinuationEnvelope, ListenExecutionContext, llm};
use serde::{Deserialize, Serialize};

use super::task::TaskStatus;

/// Typed durable per-task state owned by the task kind.
///
/// Stored on [`super::AgentTask::state`] and validated against
/// [`super::AgentTask::status`] every time the row passes through
/// [`super::AgentTask::validate`]. The default is [`TaskState::None`]
/// so freshly constructed and never-paused rows pay zero per-row
/// metadata.
///
/// `TaskState` deliberately does **not** derive `PartialEq` / `Eq`:
/// the embedded [`ContinuationEnvelope`] and [`ListenExecutionContext`]
/// payloads come from `agent-sdk-core` types that do not impl
/// `PartialEq`, and we keep equality concerns out of the schema layer.
/// Tests that need to assert `TaskState` shape compare via JSON
/// serialization (the durable wire form is the canonical contract
/// anyway).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskState {
    /// No paused state. Used by every row that is not currently in
    /// [`TaskStatus::WaitingOnChildren`] or
    /// [`TaskStatus::AwaitingConfirmation`].
    #[default]
    None,

    /// Durable state for a root task that has paused on
    /// [`TaskStatus::WaitingOnChildren`] until every spawned child
    /// reaches a terminal state.
    ///
    /// Carries the [`ContinuationEnvelope`] the worker needs to rejoin
    /// the agent loop on resume — this is the same versioned envelope
    /// the SDK already persists for `TurnOutcome::PendingToolCalls`,
    /// kept on the task row so a worker that picks the parent up after
    /// a crash can resume without reading any side-channel state.
    WaitingOnChildren {
        /// Versioned continuation envelope captured at the moment the
        /// parent paused. Wrapped in `Box` so the enum stays small and
        /// matches [`agent_sdk_core::AgentRunState::AwaitingConfirmation`].
        continuation: Box<ContinuationEnvelope>,
        /// Messages buffered at the suspension point but not yet
        /// committed: the user prompt + the assistant response
        /// (including tool-use blocks). Stored here so the resume path
        /// can reconstruct the full conversation without reading
        /// side-channel state.
        #[serde(default)]
        suspended_messages: Vec<llm::Message>,
        /// IDs of the child tasks spawned in this suspension batch.
        /// Used by the aggregation path to read only the children that
        /// belong to the current round, avoiding collisions with
        /// children from prior suspension rounds.
        #[serde(default)]
        child_ids: Vec<super::task::AgentTaskId>,
    },

    /// Durable state for a task that has paused on
    /// [`TaskStatus::AwaitingConfirmation`] until a user decision
    /// arrives.
    ///
    /// Carries both the [`ContinuationEnvelope`] (so the agent loop
    /// can resume from the exact pre-pause point) and an optional
    /// prepared [`ListenExecutionContext`] for tools that follow the
    /// listen / execute pattern. The prepared operation is `None` for
    /// non-listen tools where the resume path simply continues the
    /// loop.
    AwaitingConfirmation {
        /// Versioned continuation envelope captured at the moment the
        /// task paused on the confirmation. Boxed for the same reason
        /// as the [`TaskState::WaitingOnChildren`] payload.
        continuation: Box<ContinuationEnvelope>,
        /// Prepared listen/execute operation, when the awaited tool
        /// uses the listen tier. The resume path either executes or
        /// cancels this operation depending on the user's decision.
        ///
        /// `None` for non-listen tools (the common case).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prepared_operation: Option<ListenExecutionContext>,
    },

    /// Durable state for a task whose tool-runtime children have all
    /// reached terminal states and is ready for the resume path.
    ///
    /// Created by [`super::AgentTask::recompute_pending_children`]
    /// (Phase 4.5) when the live-child count hits zero. Carries the
    /// [`ContinuationEnvelope`] and suspended messages from the
    /// preceding [`TaskState::WaitingOnChildren`] payload so the
    /// worker that acquires the now-`Pending` parent can rebuild the
    /// conversation and call the LLM again.
    ///
    /// Unlike [`TaskState::None`], this variant is **not** a "blank
    /// slate" — the worker must inspect it and branch into the resume
    /// path rather than starting a fresh turn.
    ///
    /// The state is valid on both [`TaskStatus::Pending`] (waiting to
    /// be re-acquired) and [`TaskStatus::Running`] (already acquired
    /// and actively resuming). The canonical durable form is
    /// [`TaskStatus::Pending`]; acquisition preserves the payload while
    /// flipping the row to `Running`.
    ReadyToResume {
        /// Versioned continuation envelope from the original
        /// suspension. Contains pending tool calls, usage, and agent
        /// state at the suspension point.
        continuation: Box<ContinuationEnvelope>,
        /// Messages buffered at the original suspension point: the
        /// user prompt and the assistant response (including
        /// tool-use blocks). The resume path appends tool-result
        /// messages and rebuilds the full conversation from these.
        #[serde(default)]
        suspended_messages: Vec<llm::Message>,
        /// IDs of the child tasks that belong to this suspension
        /// batch, carried forward from the preceding
        /// [`TaskState::WaitingOnChildren`] payload so the
        /// aggregation path reads only the current round's children.
        #[serde(default)]
        child_ids: Vec<super::task::AgentTaskId>,
    },
}

impl TaskState {
    /// Returns the canonical [`TaskStatus`] this state is paired with,
    /// or `None` if the variant is compatible with any non-paused
    /// status.
    ///
    /// Used by [`super::AgentTask::validate`] to enforce the
    /// state ↔ status invariant in a single place.
    #[must_use]
    pub const fn required_status(&self) -> Option<TaskStatus> {
        match self {
            Self::None => None,
            Self::WaitingOnChildren { .. } => Some(TaskStatus::WaitingOnChildren),
            Self::AwaitingConfirmation { .. } => Some(TaskStatus::AwaitingConfirmation),
            Self::ReadyToResume { .. } => Some(TaskStatus::Pending),
        }
    }

    /// Returns `true` when the state may legally coexist with `status`.
    ///
    /// [`TaskState::ReadyToResume`] is the only non-terminal payload
    /// that intentionally spans two statuses: `Pending` while the row
    /// waits in the runnable queue and `Running` after acquisition
    /// while the worker resumes the suspended turn.
    #[must_use]
    pub const fn is_compatible_with_status(&self, status: TaskStatus) -> bool {
        match self {
            Self::None => !matches!(
                status,
                TaskStatus::WaitingOnChildren | TaskStatus::AwaitingConfirmation
            ),
            Self::WaitingOnChildren { .. } => matches!(status, TaskStatus::WaitingOnChildren),
            Self::AwaitingConfirmation { .. } => {
                matches!(status, TaskStatus::AwaitingConfirmation)
            }
            Self::ReadyToResume { .. } => {
                matches!(status, TaskStatus::Pending | TaskStatus::Running)
            }
        }
    }

    /// `true` if this is the [`TaskState::None`] default.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Borrow the [`ContinuationEnvelope`] embedded in a paused state,
    /// if any. Returns `None` for [`TaskState::None`].
    #[must_use]
    pub fn continuation(&self) -> Option<&ContinuationEnvelope> {
        match self {
            Self::None => None,
            Self::WaitingOnChildren { continuation, .. }
            | Self::AwaitingConfirmation { continuation, .. }
            | Self::ReadyToResume { continuation, .. } => Some(continuation),
        }
    }

    /// Borrow the suspended messages embedded in a
    /// [`TaskState::WaitingOnChildren`] or [`TaskState::ReadyToResume`]
    /// payload, if any. Returns an empty slice for
    /// [`TaskState::None`] and [`TaskState::AwaitingConfirmation`].
    #[must_use]
    pub fn suspended_messages(&self) -> &[llm::Message] {
        match self {
            Self::WaitingOnChildren {
                suspended_messages, ..
            }
            | Self::ReadyToResume {
                suspended_messages, ..
            } => suspended_messages,
            _ => &[],
        }
    }

    /// Borrow the child IDs embedded in a
    /// [`TaskState::WaitingOnChildren`] or [`TaskState::ReadyToResume`]
    /// payload. Returns an empty slice for all other variants.
    #[must_use]
    pub fn child_ids(&self) -> &[super::task::AgentTaskId] {
        match self {
            Self::WaitingOnChildren { child_ids, .. } | Self::ReadyToResume { child_ids, .. } => {
                child_ids
            }
            _ => &[],
        }
    }

    /// Borrow the prepared listen/execute operation embedded in an
    /// [`TaskState::AwaitingConfirmation`] payload, if any.
    ///
    /// Returns `None` for [`TaskState::None`],
    /// [`TaskState::WaitingOnChildren`], [`TaskState::ReadyToResume`],
    /// and confirmations that did not stage a listen-tier operation.
    #[must_use]
    pub const fn prepared_operation(&self) -> Option<&ListenExecutionContext> {
        match self {
            Self::AwaitingConfirmation {
                prepared_operation: Some(op),
                ..
            } => Some(op),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_core::{
        AgentContinuation, AgentState, CONTINUATION_VERSION, ContinuationEnvelope, ThreadId,
        TokenUsage,
    };

    fn sample_continuation() -> ContinuationEnvelope {
        let thread = ThreadId::from_string("t-state");
        ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        })
    }

    fn sample_prepared() -> ListenExecutionContext {
        ListenExecutionContext {
            operation_id: "op-1".into(),
            revision: 7,
            snapshot: serde_json::json!({"name": "preview"}),
            expires_at: None,
        }
    }

    #[test]
    fn default_is_none() {
        let state = TaskState::default();
        assert!(state.is_none());
        assert_eq!(state.required_status(), None);
        assert!(state.continuation().is_none());
        assert!(state.prepared_operation().is_none());
    }

    #[test]
    fn waiting_on_children_state_pins_status_and_exposes_continuation() {
        let envelope = sample_continuation();
        let state = TaskState::WaitingOnChildren {
            continuation: Box::new(envelope.clone()),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
        };
        assert_eq!(state.required_status(), Some(TaskStatus::WaitingOnChildren));
        let inner = state.continuation().expect("continuation present");
        assert_eq!(inner.version, CONTINUATION_VERSION);
        assert_eq!(inner.payload.thread_id, envelope.payload.thread_id);
        assert!(state.prepared_operation().is_none());
    }

    #[test]
    fn awaiting_confirmation_state_pins_status_and_exposes_payload() {
        let envelope = sample_continuation();
        let prepared = sample_prepared();
        let state = TaskState::AwaitingConfirmation {
            continuation: Box::new(envelope),
            prepared_operation: Some(prepared.clone()),
        };
        assert_eq!(
            state.required_status(),
            Some(TaskStatus::AwaitingConfirmation)
        );
        let op = state.prepared_operation().expect("prepared op present");
        assert_eq!(op.operation_id, prepared.operation_id);
        assert_eq!(op.revision, prepared.revision);
        assert!(state.continuation().is_some());
    }

    #[test]
    fn awaiting_confirmation_state_without_prepared_op_has_none() {
        let envelope = sample_continuation();
        let state = TaskState::AwaitingConfirmation {
            continuation: Box::new(envelope),
            prepared_operation: None,
        };
        assert!(state.prepared_operation().is_none());
        assert!(state.continuation().is_some());
    }

    #[test]
    fn task_state_round_trips_through_json_for_every_variant() -> anyhow::Result<()> {
        // None — the default for every active and terminal row.
        let none = TaskState::None;
        let json = serde_json::to_string(&none)?;
        assert_eq!(json, r#"{"kind":"none"}"#);
        let recovered: TaskState = serde_json::from_str(&json)?;
        assert!(recovered.is_none());

        // WaitingOnChildren.
        let waiting = TaskState::WaitingOnChildren {
            continuation: Box::new(sample_continuation()),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
        };
        let json = serde_json::to_string(&waiting)?;
        let recovered: TaskState = serde_json::from_str(&json)?;
        // Compare via the canonical JSON wire form because TaskState
        // does not impl PartialEq.
        assert_eq!(
            serde_json::to_value(&recovered)?,
            serde_json::to_value(&waiting)?
        );
        assert_eq!(
            recovered.required_status(),
            Some(TaskStatus::WaitingOnChildren)
        );

        // AwaitingConfirmation with a prepared listen/execute op.
        let awaiting = TaskState::AwaitingConfirmation {
            continuation: Box::new(sample_continuation()),
            prepared_operation: Some(sample_prepared()),
        };
        let json = serde_json::to_string(&awaiting)?;
        let recovered: TaskState = serde_json::from_str(&json)?;
        assert_eq!(
            serde_json::to_value(&recovered)?,
            serde_json::to_value(&awaiting)?
        );
        assert_eq!(
            recovered.required_status(),
            Some(TaskStatus::AwaitingConfirmation)
        );
        assert!(recovered.prepared_operation().is_some());

        // AwaitingConfirmation without a prepared op — must omit the
        // field on the wire so legacy readers see a clean shape.
        let awaiting_no_op = TaskState::AwaitingConfirmation {
            continuation: Box::new(sample_continuation()),
            prepared_operation: None,
        };
        let json = serde_json::to_string(&awaiting_no_op)?;
        assert!(
            !json.contains("prepared_operation"),
            "prepared_operation should be skipped when None: {json}"
        );
        let recovered: TaskState = serde_json::from_str(&json)?;
        assert_eq!(
            serde_json::to_value(&recovered)?,
            serde_json::to_value(&awaiting_no_op)?
        );
        assert!(recovered.prepared_operation().is_none());

        // ReadyToResume — durably persisted on Pending parents after
        // all children reach terminal states (Phase 4.5).
        let ready = TaskState::ReadyToResume {
            continuation: Box::new(sample_continuation()),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
        };
        let json = serde_json::to_string(&ready)?;
        let recovered: TaskState = serde_json::from_str(&json)?;
        assert_eq!(
            serde_json::to_value(&recovered)?,
            serde_json::to_value(&ready)?
        );
        assert_eq!(recovered.required_status(), Some(TaskStatus::Pending));
        assert!(recovered.continuation().is_some());
        Ok(())
    }

    #[test]
    fn ready_to_resume_state_accepts_pending_and_running() {
        let state = TaskState::ReadyToResume {
            continuation: Box::new(sample_continuation()),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
        };

        assert!(state.is_compatible_with_status(TaskStatus::Pending));
        assert!(state.is_compatible_with_status(TaskStatus::Running));
        assert!(!state.is_compatible_with_status(TaskStatus::Queued));
    }

    #[test]
    fn task_state_wire_format_uses_snake_case_kind_discriminator() -> anyhow::Result<()> {
        // Lock the discriminator format so renaming a variant by
        // accident fails this test loudly. The kind field is the only
        // forward-compatibility hinge once paused rows start landing
        // in durable storage.
        let none_value = serde_json::to_value(TaskState::None)?;
        assert_eq!(none_value["kind"], serde_json::json!("none"));

        let waiting_value = serde_json::to_value(TaskState::WaitingOnChildren {
            continuation: Box::new(sample_continuation()),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
        })?;
        assert_eq!(
            waiting_value["kind"],
            serde_json::json!("waiting_on_children")
        );
        assert!(waiting_value.get("continuation").is_some());

        let awaiting_value = serde_json::to_value(TaskState::AwaitingConfirmation {
            continuation: Box::new(sample_continuation()),
            prepared_operation: None,
        })?;
        assert_eq!(
            awaiting_value["kind"],
            serde_json::json!("awaiting_confirmation")
        );

        let ready_value = serde_json::to_value(TaskState::ReadyToResume {
            continuation: Box::new(sample_continuation()),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
        })?;
        assert_eq!(ready_value["kind"], serde_json::json!("ready_to_resume"));
        assert!(ready_value.get("continuation").is_some());
        Ok(())
    }
}
