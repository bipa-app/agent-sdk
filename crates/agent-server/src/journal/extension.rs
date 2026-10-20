//! The transition extension hook (design §6.4).
//!
//! A host registers one [`TransitionExtension`] on its store. The store
//! calls it once per [`DurableStore::commit`](super::DurableStore::commit)
//! and once per completed-turn commit, before the transaction opens, and
//! applies the writes it returns in the same transaction as the rest. The
//! extension is a pure function of what is being committed: it reads
//! nothing and does no I/O, so a retried transaction never calls it again
//! and its writes are as atomic as the writes it extends.
//!
//! For example, bip books a turn's cost into a document on the turn's own
//! thread on every completed turn, in the same transaction as the turn.
//!
//! On a commit that parks a task on its children, an extension may also
//! add a [`Write::EnqueueSteeringResume`] of that task with
//! [`SteeringInput::Queued`]: the steering queued in a document on the
//! task's own thread. It cannot read the queue, so the backend reads it
//! under the commit's lock, after the park, and hands it to
//! [`TransitionExtension::take_queued_steering`], which is just as pure:
//! it turns the queue into the steering and the queue's new value, and
//! the backend writes both in the park's transaction. Nothing queued
//! means no wake, and a second application finds the task no longer
//! parked, so a retried transaction wakes it at most once.
//!
//! The same write on a settle (a subagent completion hook adds it for its
//! own parent) delivers the steering at the wave's end. When the settle
//! fans the parent in (its last child), the backend reads the queue under
//! the same lock and hands it to [`TransitionExtension::fold_queued_steering`],
//! which appends it to that child's stored result, the one the resumed
//! parent reads; the result and the queue's new value commit with the
//! settle. A settle that does not fan the parent in wakes it, as a park
//! does, if it is still parked. Neither ever refuses the settle for a
//! steering reason: nothing queued, a declined fold or a fold that is not a
//! tool result leaves the settle as it is.
//!
//! The hook runs only on these two paths. The older single-purpose store
//! methods (`complete_task`, `cancel_tree`, `pause_on_question`, …) do not
//! call it; a host whose rule keys on a write must make sure that write
//! goes through `commit`.

use agent_sdk_foundation::ToolResult;
use agent_sdk_foundation::llm::ContentBlock;

use super::commit::CompletedTurnCommit;
use super::document::{DocExpect, Document, DocumentKey, DocumentWrite};
use super::task::{AgentTask, AgentTaskId, TaskStatus};
use super::transition::{Rejected, SteeringInput, Transition, Write};

/// What an extension sees.
#[derive(Clone, Copy, Debug)]
pub enum TransitionView<'a> {
    /// A transition on its way to [`super::DurableStore::commit`]. The
    /// extension may add [`Write::Document`] and [`Write::AppendEvents`],
    /// and a [`SteeringInput::Queued`] steering wake of a task the
    /// transition parks.
    Commit(&'a Transition),
    /// A completed turn on its way to
    /// [`super::AgentTaskStore::commit_completed_turn`]. The extension may
    /// add [`Write::Document`] only.
    CompletedTurn(&'a CompletedTurnCommit),
}

/// Adds writes to what a store commits. See the module docs.
pub trait TransitionExtension: Send + Sync {
    /// The writes to add to `view`, applied after its own writes. Empty
    /// when the extension has nothing to add.
    fn extend(&self, view: TransitionView<'_>) -> Vec<Write>;

    /// Take the steering queued for `parent`, which its transition has just
    /// parked, from `queued`, the live document a
    /// [`SteeringInput::Queued`] names, as read under the commit's lock.
    /// `None` when nothing is queued: the parent stays parked and the
    /// document is left alone. A backend may call this more than once for
    /// one commit (a retried transaction), and for a transaction that then
    /// rolls back, so it must be a pure function of its arguments.
    fn take_queued_steering(
        &self,
        parent: &AgentTask,
        queued: &Document,
    ) -> Option<QueuedSteering> {
        let _ = (parent, queued);
        None
    }

    /// Fold the steering queued for `parent` into `result`, the stored
    /// result of `settled`, the child whose settle has just fanned `parent`
    /// in (its last child, so `parent` is `Pending` with an empty
    /// [`ReadyToResume`](super::TaskState::ReadyToResume)). `queued` is the
    /// live document a [`SteeringInput::Queued`] names, read under the
    /// settle's lock. `settled.spawn_index` names the parent's pending call
    /// the result answers.
    ///
    /// Returns the result the parent reads instead, which must decode as a
    /// [`ToolResult`], and the queue's new value; both commit with the
    /// settle. `None` folds nothing and leaves the queue alone. Like
    /// [`Self::take_queued_steering`] it must be a pure function of its
    /// arguments: a backend may call it again for a retried transaction.
    fn fold_queued_steering(
        &self,
        parent: &AgentTask,
        settled: &AgentTask,
        result: &serde_json::Value,
        queued: &Document,
    ) -> Option<FoldedSteering> {
        let _ = (parent, settled, result, queued);
        None
    }
}

/// The result a fan-in settle's child carries with the steering folded in,
/// and what the queue holds after it.
#[derive(Clone, Debug)]
pub struct FoldedSteering {
    /// The settled child's new stored result. It must decode as a
    /// [`ToolResult`]; otherwise nothing is folded.
    pub result: serde_json::Value,
    /// The queue's new value, written at its next revision in the same
    /// transaction.
    pub schema_version: u32,
    pub value: serde_json::Value,
}

/// The steering an extension took from a queue, and what the queue holds
/// after it.
#[derive(Clone, Debug)]
pub struct QueuedSteering {
    /// The steering exchange's input. Empty takes nothing.
    pub steering: Vec<ContentBlock>,
    /// The queue's new value, written at its next revision in the same
    /// transaction.
    pub schema_version: u32,
    pub value: serde_json::Value,
}

/// A write a transition extension may not add; the commit is refused and
/// nothing is written.
#[derive(Debug, thiserror::Error)]
#[error("a transition extension may not add {write} to a {point}: {reason}")]
pub struct ExtensionWriteRefused {
    /// `commit` or `completed_turn`.
    pub point: &'static str,
    /// The write's [`Write::kind_name`].
    pub write: &'static str,
    pub reason: &'static str,
}

// The `outcome` of `agent_server.journal.steering_queue.takes`; the
// observability module's `attrs` names the same values.
const OUTCOME_TAKEN: &str = "taken";
const OUTCOME_EMPTY: &str = "empty";
const OUTCOME_NOT_PARKED: &str = "not_parked";
const OUTCOME_NO_EXTENSION: &str = "no_extension";
const OUTCOME_FOLDED: &str = "folded";
const OUTCOME_FOLD_EMPTY: &str = "fold_empty";
const OUTCOME_FOLD_REFUSED: &str = "fold_refused";

const POINT_COMMIT: &str = "commit";
const POINT_COMPLETED_TURN: &str = "completed_turn";

/// Why an extension may not add `write` to `transition`, or `None` when it
/// may.
fn commit_refusal(transition: &Transition, write: &Write) -> Option<&'static str> {
    match write {
        Write::Document(_) | Write::AppendEvents { .. } => None,
        Write::EnqueueSteeringResume {
            steering: SteeringInput::Blocks(_),
            ..
        } => Some("its steering wake must take its steering from a queue"),
        Write::EnqueueSteeringResume {
            parent,
            steering: SteeringInput::Queued { .. },
        } => (!parks(transition, parent)).then_some("it may wake only a task the transition parks"),
        Write::SettleTask { .. }
        | Write::RejectConfirmation { .. }
        | Write::PauseOnConfirmation { .. }
        | Write::ParkInvocation { .. }
        | Write::WakeInvocation { .. }
        | Write::ReparkAfterSteering { .. }
        | Write::Effect(_)
        | Write::AppendCompaction { .. }
        | Write::SpawnChildren { .. }
        | Write::BeginEpoch(_)
        | Write::TurnAttempt(_) => {
            Some("it may add only documents, events and a queued steering wake of a task it parks")
        }
    }
}

/// Whether `transition` parks `task` on its children: a spawn under it, or
/// its re-park after a steering exchange.
fn parks(transition: &Transition, task: &AgentTaskId) -> bool {
    transition.writes.iter().any(|write| match write {
        Write::SpawnChildren { parent, .. } => parent == task,
        Write::ReparkAfterSteering { task: parked, .. } => parked == task,
        _ => false,
    })
}

/// Apply-time half of a [`SteeringInput::Queued`] wake, shared by the
/// backends: `parent` is the task row as the transition's earlier writes
/// left it, and `stored` the live document at `queue` (the queue on
/// `parent`'s thread, [`SteeringInput::queue_on`]) as read under the
/// transition's lock. Returns the steering to wake `parent` with and the
/// queue's write, or `None` (no wake, no write) when `parent` is not
/// parked or nothing is queued. `backend` labels the count.
///
/// # Errors
/// When the store has no extension to take the queue.
#[doc(hidden)]
pub fn take_queued_steering(
    backend: &'static str,
    extension: Option<&dyn TransitionExtension>,
    parent: &AgentTask,
    queue: DocumentKey,
    stored: Option<&Document>,
) -> anyhow::Result<Option<(Vec<ContentBlock>, DocumentWrite)>> {
    let (outcome, taken) = match extension {
        None => (
            OUTCOME_NO_EXTENSION,
            Err(anyhow::anyhow!(
                "steering wake rejected: no transition extension to take the steering queued in {queue}"
            )),
        ),
        Some(_) if parent.status != TaskStatus::WaitingOnChildren => (OUTCOME_NOT_PARKED, Ok(None)),
        Some(extension) => take(extension, parent, queue, stored),
    };
    #[cfg(feature = "otel")]
    crate::observability::ServerMetrics::global().record_steering_queue(backend, outcome);
    let _ = (backend, outcome);
    taken
}

type Taken = anyhow::Result<Option<(Vec<ContentBlock>, DocumentWrite)>>;

fn take(
    extension: &dyn TransitionExtension,
    parent: &AgentTask,
    queue: DocumentKey,
    stored: Option<&Document>,
) -> (&'static str, Taken) {
    let Some(stored) = stored else {
        return (OUTCOME_EMPTY, Ok(None));
    };
    match extension.take_queued_steering(parent, stored) {
        Some(taken) if !taken.steering.is_empty() => {
            let write = DocumentWrite::put(
                queue,
                taken.schema_version,
                DocExpect::Revision(stored.revision),
                taken.value,
            );
            (OUTCOME_TAKEN, Ok(Some((taken.steering, write))))
        }
        Some(_) | None => (OUTCOME_EMPTY, Ok(None)),
    }
}

/// Apply-time half of a [`SteeringInput::Queued`] write on a settle that
/// fanned `parent` in, shared by the backends: `settled` is the child row
/// as that settle left it, and `stored` the live document at `queue` as
/// read under the transition's lock. Returns the child's new stored result
/// and the queue's write, or `None` (no write at all) when nothing is
/// queued, the extension declines, the child has no stored result or the
/// fold is not a [`ToolResult`]. `backend` labels the count.
///
/// # Errors
/// When the store has no extension to fold the queue, the refusal a
/// queued wake gives too.
#[doc(hidden)]
pub fn fold_queued_steering(
    backend: &'static str,
    extension: Option<&dyn TransitionExtension>,
    parent: &AgentTask,
    settled: &AgentTask,
    queue: DocumentKey,
    stored: Option<&Document>,
) -> anyhow::Result<Option<(serde_json::Value, DocumentWrite)>> {
    let (outcome, folded) = match extension {
        None => (
            OUTCOME_NO_EXTENSION,
            Err(anyhow::anyhow!(
                "steering fold rejected: no transition extension to fold the steering queued \
                 in {queue}"
            )),
        ),
        Some(extension) => fold(extension, parent, settled, queue, stored),
    };
    #[cfg(feature = "otel")]
    crate::observability::ServerMetrics::global().record_steering_queue(backend, outcome);
    let _ = (backend, outcome);
    folded
}

type Folded = anyhow::Result<Option<(serde_json::Value, DocumentWrite)>>;

fn fold(
    extension: &dyn TransitionExtension,
    parent: &AgentTask,
    settled: &AgentTask,
    queue: DocumentKey,
    stored: Option<&Document>,
) -> (&'static str, Folded) {
    let Some(result) = settled.result_payload.as_ref() else {
        return (OUTCOME_FOLD_REFUSED, Ok(None));
    };
    let Some(stored) = stored else {
        return (OUTCOME_FOLD_EMPTY, Ok(None));
    };
    let Some(folded) = extension.fold_queued_steering(parent, settled, result, stored) else {
        return (OUTCOME_FOLD_EMPTY, Ok(None));
    };
    if serde_json::from_value::<ToolResult>(folded.result.clone()).is_err() {
        return (OUTCOME_FOLD_REFUSED, Ok(None));
    }
    let write = DocumentWrite::put(
        queue,
        folded.schema_version,
        DocExpect::Revision(stored.revision),
        folded.value,
    );
    (OUTCOME_FOLDED, Ok(Some((folded.result, write))))
}

/// Which parents a transition's settles fan in, for its
/// [`SteeringInput::Queued`] writes. A backend reads the status of each
/// [`Self::watched`] parent just before it applies a settle, then
/// [`Self::record`]s the settle; a queued write whose parent a settle
/// fanned in folds (see [`fold_queued_steering`]) instead of waking.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct FanIns(Vec<(AgentTaskId, AgentTaskId)>);

impl FanIns {
    /// The parents of `transition`'s queued steering writes.
    pub fn watched(transition: &Transition) -> impl Iterator<Item = &AgentTaskId> {
        transition.writes.iter().filter_map(|write| match write {
            Write::EnqueueSteeringResume {
                parent,
                steering: SteeringInput::Queued { .. },
            } => Some(parent),
            _ => None,
        })
    }

    /// Record a settle: `settled` is its child and the child's parent after
    /// it, and `parked` the watched parents that were waiting on their
    /// children just before it. The settle fanned the parent in when the
    /// parent was parked, is the child's own parent, and is now `Pending`
    /// with a [`ReadyToResume`](super::TaskState::ReadyToResume).
    pub fn record(&mut self, parked: &[AgentTaskId], settled: &(AgentTask, Option<AgentTask>)) {
        let (child, Some(parent)) = settled else {
            return;
        };
        if child.parent_id.as_ref() == Some(&parent.id)
            && parked.contains(&parent.id)
            && parent.status == TaskStatus::Pending
            && matches!(parent.state, super::TaskState::ReadyToResume { .. })
        {
            self.0.push((parent.id.clone(), child.id.clone()));
        }
    }

    /// The child whose settle in this transaction fanned `parent` in.
    #[must_use]
    pub fn child_of(&self, parent: &AgentTaskId) -> Option<&AgentTaskId> {
        self.0
            .iter()
            .find_map(|(fanned, child)| (fanned == parent).then_some(child))
    }
}

impl Transition {
    /// This transition with `extension`'s writes appended. The store calls
    /// this once per commit; a host never does.
    ///
    /// # Errors
    /// [`Rejected::Store`] when the extension returns a write it may not
    /// add; nothing is committed.
    #[doc(hidden)]
    pub fn extended(
        mut self,
        extension: Option<&dyn TransitionExtension>,
    ) -> Result<Self, Rejected> {
        let Some(extension) = extension else {
            return Ok(self);
        };
        let added = extension.extend(TransitionView::Commit(&self));
        let refused = added.iter().find_map(|write| {
            commit_refusal(&self, write).map(|reason| ExtensionWriteRefused {
                point: POINT_COMMIT,
                write: write.kind_name(),
                reason,
            })
        });
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_extension(
            crate::observability::attrs::EXTENSION_POINT_COMMIT,
            added.len(),
            refused.as_ref().map(|refused| refused.write),
        );
        if let Some(refused) = refused {
            return Err(Rejected::Store(refused.into()));
        }
        self.writes.extend(added);
        Ok(self)
    }
}

impl CompletedTurnCommit {
    /// This commit with `extension`'s writes appended to its
    /// [`extra_writes`](Self::extra_writes). The store calls this once per
    /// commit; a host never does.
    ///
    /// # Errors
    /// When the extension returns a write other than a document on the
    /// turn's own thread or task; nothing is committed, and the refusal is
    /// counted.
    #[doc(hidden)]
    pub fn extended(mut self, extension: Option<&dyn TransitionExtension>) -> anyhow::Result<Self> {
        let Some(extension) = extension else {
            return Ok(self);
        };
        let added = extension.extend(TransitionView::CompletedTurn(&self));
        let refused = added
            .iter()
            .find_map(|write| Some((write, self.check_extension_write(write).err()?)));
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_extension(
            crate::observability::attrs::EXTENSION_POINT_COMPLETED_TURN,
            added.len(),
            refused.as_ref().map(|(write, _)| write.kind_name()),
        );
        if let Some((_, refused)) = refused {
            return Err(refused);
        }
        self.extra_writes.extend(added);
        Ok(self)
    }

    /// Check a write an extension adds to this completed turn: only a
    /// document, on the turn's own thread or task.
    fn check_extension_write(&self, write: &Write) -> anyhow::Result<()> {
        match write {
            Write::Document(document) => self.check_own_document(document),
            Write::SettleTask { .. }
            | Write::RejectConfirmation { .. }
            | Write::PauseOnConfirmation { .. }
            | Write::ParkInvocation { .. }
            | Write::WakeInvocation { .. }
            | Write::ReparkAfterSteering { .. }
            | Write::Effect(_)
            | Write::AppendCompaction { .. }
            | Write::AppendEvents { .. }
            | Write::SpawnChildren { .. }
            | Write::EnqueueSteeringResume { .. }
            | Write::BeginEpoch(_)
            | Write::TurnAttempt(_) => Err(ExtensionWriteRefused {
                point: POINT_COMPLETED_TURN,
                write: write.kind_name(),
                reason: "it may add only documents on the turn's own thread or task",
            }
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::document::{DocKind, DocScope};
    use crate::journal::task::{LeaseId, WorkerId};
    use agent_sdk_foundation::ThreadId;

    fn queued(parent: &AgentTaskId) -> anyhow::Result<Write> {
        Ok(Write::EnqueueSteeringResume {
            parent: parent.clone(),
            steering: SteeringInput::Queued {
                kind: DocKind::new("test.steering")?,
                key: String::new(),
            },
        })
    }

    fn repark(task: &AgentTaskId) -> Transition {
        let thread = ThreadId::from_string("t");
        let payload = crate::journal::task::SuspensionPayload {
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
                    thread_id: thread.clone(),
                    turn: 1,
                    total_usage: agent_sdk_foundation::TokenUsage::default(),
                    turn_usage: agent_sdk_foundation::TokenUsage::default(),
                    pending_tool_calls: Vec::new(),
                    awaiting_index: 0,
                    completed_results: Vec::new(),
                    state: agent_sdk_foundation::AgentState::new(thread),
                    response_id: None,
                    stop_reason: None,
                    response_content: Vec::new(),
                },
            ),
            suspended_messages: Vec::new(),
            child_join_policy: crate::ChildJoinPolicy::default(),
        };
        Transition {
            guards: Vec::new(),
            writes: vec![Write::ReparkAfterSteering {
                task: task.clone(),
                worker: WorkerId::new(),
                lease: LeaseId::new(),
                payload: Box::new(payload),
                reattach: Vec::new(),
            }],
        }
    }

    #[test]
    fn an_extension_may_wake_only_the_task_its_transition_parks() -> anyhow::Result<()> {
        let parked = AgentTaskId::new();
        let transition = repark(&parked);
        assert_eq!(commit_refusal(&transition, &queued(&parked)?), None);
        assert!(commit_refusal(&transition, &queued(&AgentTaskId::new())?).is_some());
        assert!(commit_refusal(&Transition::default(), &queued(&parked)?).is_some());
        let inline = Write::EnqueueSteeringResume {
            parent: parked,
            steering: vec![ContentBlock::Text { text: "x".into() }].into(),
        };
        assert!(commit_refusal(&transition, &inline).is_some());
        Ok(())
    }

    /// A settle fans in only the settled child's own parent, one that was
    /// parked just before it and is now `Pending` to resume. A linked
    /// child-thread root (no `parent_id`) reports the invocation it resumes
    /// as its parent; that is not a fan-in even in a resume shape.
    #[test]
    fn a_settle_fans_in_only_its_own_parked_parent() -> anyhow::Result<()> {
        let now = time::OffsetDateTime::UNIX_EPOCH;
        let mut parent = AgentTask::new_root_turn(ThreadId::from_string("t"), now, 3);
        parent.status = TaskStatus::Pending;
        parent.state = crate::journal::TaskState::ReadyToResume {
            continuation: Box::new(continuation()?),
            suspended_messages: Vec::new(),
            child_ids: Vec::new(),
            child_join_policy: crate::ChildJoinPolicy::default(),
            steering: Vec::new(),
        };
        let mut child = AgentTask::new_root_turn(ThreadId::from_string("t"), now, 3);
        child.parent_id = Some(parent.id.clone());
        let linked = AgentTask::new_root_turn(ThreadId::from_string("c"), now, 3);
        let parked = [parent.id.clone()];

        let mut fan_ins = FanIns::default();
        fan_ins.record(&parked, &(linked, Some(parent.clone())));
        assert_eq!(fan_ins.child_of(&parent.id), None, "not its own parent");
        fan_ins.record(&[], &(child.clone(), Some(parent.clone())));
        assert_eq!(fan_ins.child_of(&parent.id), None, "not parked before");
        fan_ins.record(&parked, &(child.clone(), Some(parent.clone())));
        assert_eq!(fan_ins.child_of(&parent.id), Some(&child.id));
        Ok(())
    }

    /// The continuation of [`repark`]'s suspension payload.
    fn continuation() -> anyhow::Result<agent_sdk_foundation::ContinuationEnvelope> {
        match repark(&AgentTaskId::new()).writes.remove(0) {
            Write::ReparkAfterSteering { payload, .. } => Ok(payload.continuation),
            other => anyhow::bail!("repark builds a re-park, not {}", other.kind_name()),
        }
    }

    #[test]
    fn a_queued_wake_may_not_share_its_transition_with_a_write_of_its_queue() -> anyhow::Result<()>
    {
        let parked = AgentTaskId::new();
        let queue = DocumentKey::new(
            DocScope::Thread(ThreadId::from_string("t")),
            DocKind::new("test.steering")?,
            "",
        );
        let transition = Transition {
            guards: Vec::new(),
            writes: vec![
                Write::Document(DocumentWrite::put(
                    queue,
                    1,
                    DocExpect::Any,
                    serde_json::json!([]),
                )),
                queued(&parked)?,
            ],
        };
        assert!(matches!(transition.validate(), Err(Rejected::Store(_))));
        Ok(())
    }
}
