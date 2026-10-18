//! Progress events the worker writes about a task it holds: a subagent's
//! periodic `SubagentProgress` frame on its parent's thread, and a plain
//! root turn's stall notice on its own thread.
//!
//! Each is appended under the lease of the task the emitter runs
//! ([`Transition::append_owned`]), so an emitter that lost its lease writes
//! nothing, and neither does one whose target thread is purged. The commit
//! goes straight to the store, never through the activity-tracking event
//! repository: reporting progress is not evidence of work, so a child that
//! only reports progress still stalls.

use agent_sdk_foundation::{
    AgentEvent, PendingToolCallInfo, SubagentLiveness, SubagentPhase, ThreadId, TokenUsage,
};
use agent_server::journal::Rejected;
use agent_server::journal::task::{
    AgentTask, AgentTaskId, LeaseId, TaskKind, TaskStatus, WorkerId,
};
use agent_server::journal::thread::ThreadStateConflict;
use agent_server::journal::transition::Guard;
use agent_server::journal::transition::Transition;
use agent_server::journal::turn_attempt::TurnAttempt;
use agent_server::worker::{
    ActivityBeacon, SubagentProgressSnapshot, build_parent_progress_event, canonical_subagent_name,
};
use anyhow::{Context, Result};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::publish_events;
use crate::stores::StoreRegistry;

/// The task an emitter runs, and the lease it holds on it.
#[derive(Clone, Debug)]
pub(super) struct LeaseHolder {
    pub(super) task: AgentTaskId,
    pub(super) worker: WorkerId,
    pub(super) lease: LeaseId,
}

impl LeaseHolder {
    pub(super) fn new(task: &AgentTaskId, worker: &WorkerId, lease: &LeaseId) -> Self {
        Self {
            task: task.clone(),
            worker: worker.clone(),
            lease: lease.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ProgressKind {
    SubagentProgress,
    RootStallNotice,
}

impl ProgressKind {
    #[cfg(feature = "otel")]
    const fn label(self) -> &'static str {
        use agent_server::observability::attrs;
        match self {
            Self::SubagentProgress => attrs::PROGRESS_SUBAGENT,
            Self::RootStallNotice => attrs::PROGRESS_ROOT_STALL_NOTICE,
        }
    }
}

/// Why a progress append wrote nothing.
#[derive(Debug)]
pub(super) enum Refusal {
    /// The emitter's task is no longer running under its lease.
    LeaseLost,
    /// The target thread is purged or being purged.
    Purged,
    /// The store failed; the next tick retries.
    Store(anyhow::Error),
    /// The task the frame reports on is no longer parked: another emitter
    /// reports on it now. Not counted.
    SubjectMoved,
}

impl Refusal {
    #[cfg(feature = "otel")]
    const fn label(&self) -> Option<&'static str> {
        use agent_server::observability::attrs;
        match self {
            Self::LeaseLost => Some(attrs::PROGRESS_LEASE_LOST),
            Self::Purged => Some(attrs::PROGRESS_PURGED),
            Self::Store(_) => Some(attrs::PROGRESS_STORE),
            Self::SubjectMoved => None,
        }
    }
}

/// Append `events` to `thread` while `holder` still runs its task under its
/// lease, and hand the committed events to live followers.
pub(super) async fn append_owned(
    stores: &StoreRegistry,
    holder: &LeaseHolder,
    thread: &ThreadId,
    events: Vec<AgentEvent>,
    kind: ProgressKind,
    now: OffsetDateTime,
) -> Result<(), Refusal> {
    let transition = Transition::append_owned(
        holder.task.clone(),
        holder.worker.clone(),
        holder.lease.clone(),
        thread.clone(),
        events,
    );
    commit_progress(stores, transition, kind, now).await
}

/// Commit a progress transition whose first guard is the emitter's lease.
/// A later guard that fails means the subject moved on, not a refusal.
async fn commit_progress(
    stores: &StoreRegistry,
    transition: Transition,
    kind: ProgressKind,
    now: OffsetDateTime,
) -> Result<(), Refusal> {
    let refusal = match stores.task_store.commit(transition, now).await {
        Ok(committed) => {
            publish_events(stores, &committed.events);
            #[cfg(feature = "otel")]
            agent_server::observability::ServerMetrics::global()
                .record_progress_event(kind.label());
            #[cfg(not(feature = "otel"))]
            let _ = kind;
            return Ok(());
        }
        Err(Rejected::Guard { index: 0, .. }) => Refusal::LeaseLost,
        Err(Rejected::Guard { .. }) => Refusal::SubjectMoved,
        Err(Rejected::Store(error)) if error.downcast_ref::<ThreadStateConflict>().is_some() => {
            Refusal::Purged
        }
        Err(Rejected::Store(error)) => Refusal::Store(error),
    };
    #[cfg(feature = "otel")]
    if let Some(reason) = refusal.label() {
        agent_server::observability::ServerMetrics::global().record_progress_refused(reason);
    }
    Err(refusal)
}

/// Who reports on a subagent invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Emitter {
    /// The invocation's child root, while its turn runs.
    ChildRoot,
    /// A tool child of the child root, while the root is parked on it.
    ToolChild,
}

/// The invocation an emitter reports on, named by its child root.
#[derive(Clone, Debug)]
pub(super) struct Subject {
    child_root: AgentTaskId,
    /// The child root's thread, which the emitter's own row is on too.
    child_thread: ThreadId,
    emitter: Emitter,
}

/// The subagent child root `task` reports on, if any: itself when it is a
/// linked child root, its parent when it is a tool child of one.
pub(super) async fn resolve_subject(
    stores: &StoreRegistry,
    task: &AgentTask,
) -> Result<Option<Subject>> {
    let (candidate, emitter) = match task.kind {
        TaskKind::RootTurn => (&task.id, Emitter::ChildRoot),
        TaskKind::ToolRuntime => match task.parent_id.as_ref() {
            Some(parent) => (parent, Emitter::ToolChild),
            None => return Ok(None),
        },
        TaskKind::Subagent | TaskKind::InputInjection => return Ok(None),
    };
    let linked = stores
        .task_store
        .find_subagent_invocation_for_child_root(candidate)
        .await
        .context("find the subagent invocation a progress emitter reports on")?;
    Ok(linked.map(|_| Subject {
        child_root: candidate.clone(),
        child_thread: task.thread_id.clone(),
        emitter,
    }))
}

pub(super) struct SubagentProgressParams {
    pub(super) stores: StoreRegistry,
    pub(super) task: AgentTask,
    pub(super) holder: LeaseHolder,
    pub(super) interval: std::time::Duration,
    pub(super) activity: ActivityBeacon,
    pub(super) cancel: CancellationToken,
}

/// Report a subagent invocation's progress to its parent every `interval`
/// while this acquisition holds its lease.
///
/// A linked child root reports while its turn runs. A tool child of one
/// reports on the invocation's behalf while the root is parked on it, so a
/// child waiting on a 40-minute build still shows progress; of several such
/// tool children, only one reports per tick. The loop stops when its lease
/// is lost or the parent's thread is purged.
pub(super) async fn subagent_progress_loop(params: SubagentProgressParams) {
    let SubagentProgressParams {
        stores,
        task,
        holder,
        interval,
        activity,
        cancel,
    } = params;
    // Resolved once at dispatch, so a plain root or a plain root's tool child
    // costs one query. A store error is not "no linkage": it is retried each
    // tick rather than silencing a real subagent for the whole acquisition.
    let mut subject = match resolve_subject(&stores, &task).await {
        Ok(None) => return,
        Ok(Some(subject)) => Some(subject),
        Err(err) => {
            warn!(
                task_id = %task.id,
                thread_id = %task.thread_id,
                error = %err,
                "subagent progress linkage probe failed; retrying per tick",
            );
            None
        }
    };

    let mut ticker = tokio::time::interval(interval);
    ticker.tick().await;
    loop {
        tokio::select! {
            () = cancel.cancelled() => return,
            _ = ticker.tick() => {}
        }
        if subject.is_none() {
            match resolve_subject(&stores, &task).await {
                Ok(None) => return,
                Ok(Some(resolved)) => subject = Some(resolved),
                Err(err) => {
                    warn!(task_id = %task.id, error = %err, "subagent progress linkage probe failed");
                    continue;
                }
            }
        }
        let Some(resolved) = subject.as_ref() else {
            continue;
        };
        let outcome = emit_subagent_progress(
            &stores,
            &holder,
            resolved,
            &activity,
            OffsetDateTime::now_utc(),
        )
        .await;
        match outcome {
            Ok(()) | Err(FrameError::Refused(Refusal::SubjectMoved)) => {}
            Err(FrameError::Refused(Refusal::LeaseLost)) => {
                info!(
                    task_id = %task.id,
                    "the progress emitter lost its lease; subagent progress stops",
                );
                return;
            }
            Err(FrameError::Refused(Refusal::Purged)) => {
                info!(
                    task_id = %task.id,
                    "the parent thread was purged; subagent progress stops",
                );
                return;
            }
            Err(FrameError::Refused(Refusal::Store(err)) | FrameError::Read(err)) => {
                warn!(
                    task_id = %task.id,
                    error = %err,
                    "periodic subagent progress failed; retrying next interval",
                );
            }
        }
    }
}

#[derive(Debug)]
pub(super) enum FrameError {
    Read(anyhow::Error),
    Refused(Refusal),
}

impl From<anyhow::Error> for FrameError {
    fn from(error: anyhow::Error) -> Self {
        Self::Read(error)
    }
}

/// Write one periodic progress frame for `subject`'s invocation, unless
/// another emitter is the one to report this tick.
pub(super) async fn emit_subagent_progress(
    stores: &StoreRegistry,
    holder: &LeaseHolder,
    subject: &Subject,
    activity: &ActivityBeacon,
    now: OffsetDateTime,
) -> Result<(), FrameError> {
    // Task rows before the thread projection: a turn's usage moves from the
    // child root's continuation into `total_usage` only when the turn
    // commits, and no turn commits while a lease-holding emitter reports.
    let tasks = stores
        .task_store
        .list_by_thread(&subject.child_thread)
        .await
        .context("list periodic progress child tasks")?;
    let child_root = tasks
        .iter()
        .find(|task| task.id == subject.child_root)
        .context("periodic progress child root no longer exists")?;
    match turn(holder, subject, &tasks, now) {
        Turn::Report => {}
        Turn::Skip => return Ok(()),
        Turn::LeaseLost => return Err(FrameError::Refused(Refusal::LeaseLost)),
    }
    let Some(invocation) = stores
        .task_store
        .find_subagent_invocation_for_child_root(&subject.child_root)
        .await
        .context("find invocation for periodic subagent progress")?
    else {
        return Ok(());
    };
    let linkage = invocation
        .state
        .subagent_invocation()
        .context("periodic progress invocation missing linkage")?;
    let (parent_thread, pending_tool) = spawning_call(stores, &invocation).await?;
    let child_thread = stores
        .thread_store
        .get(&linkage.child_thread_id)
        .await
        .context("load periodic progress child thread")?
        .context("periodic progress child thread no longer exists")?;
    let attempts = stores
        .attempt_store
        .list_by_task(&subject.child_root)
        .await
        .context("list periodic progress child attempts")?;

    let usage = live_usage(&child_thread.total_usage, child_root, subject, activity);
    let tool_count = u32::try_from(
        tasks
            .iter()
            .filter(|task| {
                task.kind == TaskKind::ToolRuntime
                    && matches!(task.status, TaskStatus::Completed | TaskStatus::Failed)
            })
            .count(),
    )
    .context("periodic progress tool count exceeds u32")?;
    let liveness = Live {
        subject,
        activity,
        tasks: &tasks,
        child_root,
        attempts: &attempts,
    }
    .observe(linkage.spec.timeout_ms, now);
    let event = build_parent_progress_event(&SubagentProgressSnapshot {
        subagent_id: &pending_tool.id,
        subagent_name: canonical_subagent_name(&pending_tool.name),
        spec: &linkage.spec,
        child_thread_id: &linkage.child_thread_id,
        child_root_task_id: &linkage.child_root_task_id,
        subagent_task_id: &invocation.id,
        completed: false,
        success: false,
        current_turn: child_thread.committed_turns.saturating_add(1),
        tool_count,
        usage,
        liveness: Some(liveness),
    });
    let mut transition = Transition::append_owned(
        holder.task.clone(),
        holder.worker.clone(),
        holder.lease.clone(),
        parent_thread,
        vec![event],
    );
    match subject.emitter {
        Emitter::ChildRoot => {}
        // Only while the child root is still parked: once it resumes or
        // ends, its own loop or its completion frame reports.
        Emitter::ToolChild => transition.guards.push(Guard::TaskIs {
            task: subject.child_root.clone(),
            status: TaskStatus::WaitingOnChildren,
            lease: None,
        }),
    }
    commit_progress(stores, transition, ProgressKind::SubagentProgress, now)
        .await
        .map_err(FrameError::Refused)
}

enum Turn {
    Report,
    Skip,
    LeaseLost,
}

/// The parent's thread and the pending call that spawned `invocation`, the
/// call whose id keys the parent's card.
async fn spawning_call(
    stores: &StoreRegistry,
    invocation: &AgentTask,
) -> Result<(ThreadId, PendingToolCallInfo)> {
    let parent_id = invocation
        .parent_id
        .as_ref()
        .context("periodic progress invocation missing parent")?;
    let parent = stores
        .task_store
        .get(parent_id)
        .await
        .context("load periodic progress parent")?
        .context("periodic progress parent no longer exists")?;
    let spawn_index = usize::try_from(
        invocation
            .spawn_index
            .context("periodic progress invocation missing spawn_index")?,
    )
    .context("periodic progress spawn_index exceeds usize")?;
    let pending_tool = parent
        .state
        .continuation()
        .and_then(|continuation| continuation.payload.pending_tool_calls.get(spawn_index))
        .context("periodic progress parent missing pending subagent tool")?
        .clone();
    Ok((parent.thread_id, pending_tool))
}

/// What a frame's liveness is read from.
struct Live<'a> {
    subject: &'a Subject,
    activity: &'a ActivityBeacon,
    tasks: &'a [AgentTask],
    child_root: &'a AgentTask,
    attempts: &'a [TurnAttempt],
}

impl Live<'_> {
    fn observe(&self, budget_ms: u64, now: OffsetDateTime) -> SubagentLiveness {
        let last_activity_at = self
            .activity
            .latest()
            .into_iter()
            .chain(self.tasks.iter().filter_map(|task| task.last_activity_at))
            .chain(
                self.attempts
                    .iter()
                    .map(|attempt| attempt.closed_at.unwrap_or(attempt.opened_at)),
            )
            .max();
        let silent_since = last_activity_at.unwrap_or(self.child_root.created_at);
        SubagentLiveness {
            last_activity_at,
            phase: phase(self.subject.emitter, now - silent_since, budget_ms),
            llm_calls: u64::try_from(self.attempts.len()).unwrap_or(u64::MAX),
            output_tokens_live: self
                .attempts
                .iter()
                .map(|attempt| u64::from(attempt.output_tokens.unwrap_or(0)))
                .sum(),
        }
    }
}

/// Whether `holder` is the emitter that reports `subject` this tick.
///
/// The child root reports while it runs. While it is parked, the running
/// tool child with the lowest id reports, among those whose lease has not
/// expired; a crashed sibling's row is skipped once its lease lapses. That
/// the child root is still parked is the append's own guard. An emitter
/// that no longer holds its own row stops, whether or not it would report.
fn turn(holder: &LeaseHolder, subject: &Subject, tasks: &[AgentTask], now: OffsetDateTime) -> Turn {
    let holds_its_row = tasks.iter().any(|task| {
        task.id == holder.task
            && task.status == TaskStatus::Running
            && task.lease_id.as_ref() == Some(&holder.lease)
    });
    if !holds_its_row {
        return Turn::LeaseLost;
    }
    match subject.emitter {
        Emitter::ChildRoot => Turn::Report,
        Emitter::ToolChild => {
            let elected = tasks
                .iter()
                .filter(|task| {
                    task.kind == TaskKind::ToolRuntime
                        && task.status == TaskStatus::Running
                        && task.parent_id.as_ref() == Some(&subject.child_root)
                        && task.lease_expires_at.is_none_or(|expires| expires > now)
                })
                .map(|task| &task.id)
                .min();
            if elected == Some(&holder.task) {
                Turn::Report
            } else {
                Turn::Skip
            }
        }
    }
}

/// The child's usage so far: its committed turns, the open turn's usage the
/// child root's continuation carries, and what the child root's own beacon
/// streamed since it was acquired. Each token is in exactly one of the three.
fn live_usage(
    committed: &TokenUsage,
    child_root: &AgentTask,
    subject: &Subject,
    activity: &ActivityBeacon,
) -> TokenUsage {
    let mut usage = committed.clone();
    if let Some(continuation) = child_root.state.continuation() {
        usage.add(&continuation.payload.turn_usage);
    }
    match subject.emitter {
        Emitter::ChildRoot => usage.add(&activity.usage()),
        Emitter::ToolChild => {}
    }
    usage
}

/// `Stalled` once the child has been silent for half its stall budget;
/// otherwise what the emitter says the child is doing.
fn phase(emitter: Emitter, silent_for: time::Duration, budget_ms: u64) -> SubagentPhase {
    let half_budget =
        time::Duration::milliseconds(i64::try_from(budget_ms / 2).unwrap_or(i64::MAX));
    if budget_ms > 0 && silent_for >= half_budget {
        return SubagentPhase::Stalled;
    }
    match emitter {
        Emitter::ChildRoot => SubagentPhase::Working,
        Emitter::ToolChild => SubagentPhase::WaitingOnTools,
    }
}
