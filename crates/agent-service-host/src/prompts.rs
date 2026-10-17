//! Answering, closing and re-driving prompt rows
//! ([`agent_server::journal::prompt`]).
//!
//! A host owns the prompts it opens: it registers one [`PromptHandler`]
//! with [`crate::grpc::GrpcTransport::with_prompt_handler`], and the
//! control plane calls it once a host prompt is answered. The handler
//! applies the answer and says how to close the prompt; the close commits
//! under the revision the answer left, so it lands once however often the
//! handler runs. An answered prompt that never closed (the process died, or
//! the handler asked to wait) is handed to the handler again by
//! [`redrive_answered_prompts`] when the transport starts. An open prompt
//! needs nothing running: it is a row, and the answer arrives as an RPC.
//!
//! A task-owned prompt is closed when its task stops awaiting its
//! confirmation: right after a decision through the control plane, and by
//! [`withdraw_stale_task_prompts`] for a task resolved any other way (a
//! cancelled tree, a failed root, a crash between the decision and the
//! close).
use std::sync::Arc;

use agent_sdk_foundation::{ThreadId, ToolResult};
use agent_server::journal::prompt::{
    PromptAnswer, PromptOwner, PromptRecord, PromptRefusal, PromptState, StoredPrompt, all_prompts,
    get_prompt, owner_task_awaits, step,
};
use agent_server::journal::{
    AgentTask, AgentTaskId, AgentTaskStore, Committed, DocumentStore, Rejected, Transition,
};
use async_trait::async_trait;
use time::OffsetDateTime;
use tracing::{Instrument, info_span, warn};

/// How a [`PromptHandler`] wants an answered prompt closed.
#[derive(Clone, Debug)]
pub enum PromptResolution {
    /// Close it with the pair carrying `result`. `retain` keeps the row as
    /// a tombstone, so the same id cannot open again until the host retires
    /// it. `extra` holds the host's own guards and writes (for example the
    /// wake of the prompt's owner task); they commit in the close's
    /// transaction, and a failed guard leaves the prompt answered.
    Close {
        result: ToolResult,
        retain: bool,
        extra: Transition,
    },
    /// Leave it answered: the handler could not apply it yet, and the next
    /// start hands it over again.
    Pending,
}

/// Applies the answers to a host's own prompts.
#[async_trait]
pub trait PromptHandler: Send + Sync {
    /// Apply `prompt`'s answer. It may run more than once for one answer
    /// (a crash before the close, a restart), so it must be idempotent.
    ///
    /// # Errors
    /// The answer could not be applied; the prompt stays answered.
    async fn on_answered(&self, prompt: &StoredPrompt) -> anyhow::Result<PromptResolution>;

    /// The result a task-owned prompt row closes with once its task no
    /// longer awaits (`task` is `None` once the row is gone).
    fn task_prompt_result(&self, record: &PromptRecord, task: Option<&AgentTask>) -> ToolResult {
        default_task_prompt_result(record, task)
    }
}

/// A shared handler.
pub type SharedPromptHandler = Arc<dyn PromptHandler>;

/// What answering a prompt row did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answered {
    /// This request answered it.
    Fresh(StoredPrompt),
    /// This request had already answered it.
    Replay(StoredPrompt),
}

impl Answered {
    #[must_use]
    pub const fn prompt(&self) -> &StoredPrompt {
        match self {
            Self::Fresh(prompt) | Self::Replay(prompt) => prompt,
        }
    }
}

/// Why a prompt row was not answered.
#[derive(Debug)]
pub enum AnswerError {
    NotFound {
        thread: ThreadId,
        id: String,
    },
    /// Another request answered it.
    AlreadyAnswered {
        id: String,
    },
    Closed {
        id: String,
    },
    /// A task-owned prompt: its task's decision is the answer.
    TaskOwned {
        id: String,
        task: AgentTaskId,
    },
    Store(anyhow::Error),
}

impl std::fmt::Display for AnswerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { thread, id } => write!(f, "no prompt {id} on thread {thread}"),
            Self::AlreadyAnswered { id } => {
                write!(f, "prompt {id} is already answered by another request")
            }
            Self::Closed { id } => write!(f, "prompt {id} is closed"),
            Self::TaskOwned { id, task } => {
                write!(f, "prompt {id} belongs to task {task}; decide the task")
            }
            Self::Store(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for AnswerError {}

impl From<anyhow::Error> for AnswerError {
    fn from(error: anyhow::Error) -> Self {
        Self::Store(error)
    }
}

/// Commit a transition that writes prompt rows, and count it on
/// `agent_server.prompts` as one `step` of a prompt owned by `owner`.
///
/// # Errors
/// As [`agent_server::journal::DurableStore::commit`].
pub async fn commit_prompt_step(
    store: &dyn AgentTaskStore,
    transition: Transition,
    owner: &PromptOwner,
    step: &'static str,
    now: OffsetDateTime,
) -> Result<Committed, Rejected> {
    #[cfg(feature = "otel")]
    let guard_count = transition.guards.len();
    let result = store.commit(transition, now).await;
    #[cfg(feature = "otel")]
    agent_server::observability::ServerMetrics::global().record_prompt(
        owner.label(),
        step,
        agent_server::observability::ServerMetrics::commit_outcome(guard_count, &result),
    );
    #[cfg(not(feature = "otel"))]
    let _ = (owner, step);
    result
}

fn refusal(id: &str, error: anyhow::Error) -> AnswerError {
    match error.downcast::<PromptRefusal>() {
        Ok(PromptRefusal::AlreadyAnswered { .. }) => {
            AnswerError::AlreadyAnswered { id: id.to_owned() }
        }
        Ok(PromptRefusal::Closed { .. }) => AnswerError::Closed { id: id.to_owned() },
        Err(error) => AnswerError::Store(error),
    }
}

/// Store `answer` on the host prompt `id` of `thread`, once. The same
/// request answering again is a [`Answered::Replay`]; any other request
/// gets [`AnswerError::AlreadyAnswered`].
///
/// # Errors
/// [`AnswerError`].
pub async fn answer_prompt(
    store: &dyn AgentTaskStore,
    documents: &dyn DocumentStore,
    thread: &ThreadId,
    id: &str,
    answer: PromptAnswer,
    now: OffsetDateTime,
) -> Result<Answered, AnswerError> {
    let span = info_span!("agent_server.prompt.answer", prompt.id = id, thread.id = %thread);
    async move {
        let Some(prompt) = get_prompt(documents, thread, id).await? else {
            return Err(AnswerError::NotFound {
                thread: thread.clone(),
                id: id.to_owned(),
            });
        };
        if let PromptOwner::Task { task_id, .. } = &prompt.record.owner {
            return Err(AnswerError::TaskOwned {
                id: id.to_owned(),
                task: task_id.clone(),
            });
        }
        let write = match prompt.answer_write(answer.clone()) {
            Ok(write) => write,
            Err(error) => {
                let replay = matches!(
                    error.downcast_ref::<PromptRefusal>(),
                    Some(PromptRefusal::AlreadyAnswered {
                        same_request: true,
                        ..
                    })
                );
                if replay {
                    return Ok(Answered::Replay(prompt));
                }
                return Err(refusal(id, error));
            }
        };
        let transition = Transition {
            guards: Vec::new(),
            writes: vec![write],
        };
        let owner = prompt.record.owner.clone();
        match commit_prompt_step(store, transition, &owner, step::ANSWER, now).await {
            Ok(committed) => {
                let revision = committed
                    .documents
                    .first()
                    .and_then(|outcome| outcome.revision)
                    .unwrap_or_else(|| prompt.revision.saturating_add(1));
                let mut record = prompt.record;
                record.state = PromptState::Answered { answer };
                Ok(Answered::Fresh(StoredPrompt { record, revision }))
            }
            Err(Rejected::Guard { .. }) => {
                let current = get_prompt(documents, thread, id).await?.ok_or_else(|| {
                    AnswerError::NotFound {
                        thread: thread.clone(),
                        id: id.to_owned(),
                    }
                })?;
                match current.record.state.answer() {
                    Some(stored) if stored.request_id == answer.request_id => {
                        Ok(Answered::Replay(current))
                    }
                    Some(_) => Err(AnswerError::AlreadyAnswered { id: id.to_owned() }),
                    None if current.revision == prompt.revision => Err(AnswerError::Store(
                        anyhow::anyhow!("the store refused the answer to prompt {id} (is its thread being purged?)"),
                    )),
                    None => Err(AnswerError::Closed { id: id.to_owned() }),
                }
            }
            Err(Rejected::Store(error)) => Err(AnswerError::Store(error)),
        }
    }
    .instrument(span)
    .await
}

/// Close `prompt` with `result`, with `extra` in the same transaction.
///
/// `extra` holds the caller's own guards and writes. `false` when a guard
/// failed or another writer moved the prompt first (it is then closed, or
/// answered anew, by that writer).
///
/// # Errors
/// The prompt is already closed, or the store failed.
pub async fn close_prompt(
    store: &dyn AgentTaskStore,
    prompt: &StoredPrompt,
    result: ToolResult,
    retain: bool,
    extra: Transition,
    step: &'static str,
    now: OffsetDateTime,
) -> anyhow::Result<bool> {
    let mut transition = extra;
    transition
        .writes
        .extend(prompt.close_writes(result, retain, now)?);
    match commit_prompt_step(store, transition, &prompt.record.owner, step, now).await {
        Ok(_) => Ok(true),
        Err(Rejected::Guard { .. }) => Ok(false),
        Err(Rejected::Store(error)) => Err(error),
    }
}

/// Hand an answered host prompt to `handler` and commit the close it asks
/// for. `true` once the prompt is closed.
///
/// # Errors
/// The handler or the store failed; the prompt stays answered.
pub async fn resolve_answered(
    store: &dyn AgentTaskStore,
    prompt: &StoredPrompt,
    handler: &dyn PromptHandler,
    now: OffsetDateTime,
) -> anyhow::Result<bool> {
    let span = info_span!(
        "agent_server.prompt.handle",
        prompt.id = %prompt.record.id,
        thread.id = %prompt.record.surface_thread,
    );
    async move {
        let resolution = handler.on_answered(prompt).await;
        #[cfg(feature = "otel")]
        agent_server::observability::ServerMetrics::global().record_prompt(
            prompt.record.owner.label(),
            step::HANDLE,
            match &resolution {
                Ok(PromptResolution::Close { .. }) => {
                    agent_server::observability::attrs::OUTCOME_CLOSED
                }
                Ok(PromptResolution::Pending) => {
                    agent_server::observability::attrs::OUTCOME_PENDING
                }
                Err(_) => agent_server::observability::attrs::OUTCOME_ERROR,
            },
        );
        match resolution? {
            PromptResolution::Close {
                result,
                retain,
                extra,
            } => close_prompt(store, prompt, result, retain, extra, step::CLOSE, now).await,
            PromptResolution::Pending => Ok(false),
        }
    }
    .instrument(span)
    .await
}

/// Hand every answered, unclosed host prompt to `handler`, oldest first.
/// One failing prompt does not stop the others. Returns how many closed.
///
/// # Errors
/// The prompt rows could not be read.
pub async fn redrive_answered_prompts(
    store: &dyn AgentTaskStore,
    documents: &dyn DocumentStore,
    handler: &dyn PromptHandler,
    now: OffsetDateTime,
) -> anyhow::Result<usize> {
    let mut answered = all_prompts(documents, |record| {
        matches!(record.owner, PromptOwner::Host { .. })
            && matches!(record.state, PromptState::Answered { .. })
    })
    .await?;
    answered.sort_by_key(|prompt| prompt.record.opened_at);
    let mut closed = 0;
    for prompt in &answered {
        match resolve_answered(store, prompt, handler, now).await {
            Ok(true) => closed += 1,
            Ok(false) => {}
            Err(error) => warn!(
                prompt.id = %prompt.record.id,
                thread.id = %prompt.record.surface_thread,
                error = %format!("{error:#}"),
                "answered prompt could not be applied; the next start retries it"
            ),
        }
    }
    Ok(closed)
}

/// The result a task-owned prompt closes with, given its task as it now
/// stands (`None` once the task row is gone).
pub type TaskPromptResult = dyn Fn(&PromptRecord, Option<&AgentTask>) -> ToolResult + Send + Sync;

/// The default [`TaskPromptResult`]: success when the task went on to run
/// or finish, an error naming the task's state otherwise.
#[must_use]
pub fn default_task_prompt_result(_: &PromptRecord, task: Option<&AgentTask>) -> ToolResult {
    match task.map(|task| task.status) {
        Some(
            agent_server::journal::TaskStatus::Pending
            | agent_server::journal::TaskStatus::Running
            | agent_server::journal::TaskStatus::Completed,
        ) => ToolResult::success("No longer awaiting"),
        Some(status) => ToolResult::error(format!("withdrawn: the task is {status:?}")),
        None => ToolResult::error("withdrawn: the task is gone"),
    }
}

/// Close every open task-owned prompt that `select` keeps and whose task
/// no longer awaits its confirmation. Returns how many closed.
async fn close_stale_task_prompts(
    tasks: &dyn AgentTaskStore,
    documents: &dyn DocumentStore,
    select: impl Fn(&AgentTaskId) -> bool + Send + Sync,
    result: &TaskPromptResult,
    now: OffsetDateTime,
) -> anyhow::Result<usize> {
    let open = all_prompts(documents, |record| {
        matches!(record.state, PromptState::Open)
            && matches!(&record.owner, PromptOwner::Task { task_id, .. } if select(task_id))
    })
    .await?;
    let mut closed = 0;
    for prompt in &open {
        if owner_task_awaits(tasks, &prompt.record).await? {
            continue;
        }
        let PromptOwner::Task { task_id, .. } = &prompt.record.owner else {
            continue;
        };
        let task = tasks.get(task_id).await?;
        if close_prompt(
            tasks,
            prompt,
            result(&prompt.record, task.as_ref()),
            false,
            Transition::default(),
            step::WITHDRAW,
            now,
        )
        .await?
        {
            closed += 1;
        }
    }
    Ok(closed)
}

/// Close the open prompts owned by `task` once it no longer awaits its
/// confirmation.
///
/// # Errors
/// A store failed.
pub async fn close_task_prompts(
    tasks: &dyn AgentTaskStore,
    documents: &dyn DocumentStore,
    task: &AgentTaskId,
    result: &TaskPromptResult,
    now: OffsetDateTime,
) -> anyhow::Result<usize> {
    close_stale_task_prompts(tasks, documents, |owner| owner == task, result, now).await
}

/// Close every open task-owned prompt whose task no longer awaits its
/// confirmation, whatever resolved it.
///
/// # Errors
/// A store failed.
pub async fn withdraw_stale_task_prompts(
    tasks: &dyn AgentTaskStore,
    documents: &dyn DocumentStore,
    result: &TaskPromptResult,
    now: OffsetDateTime,
) -> anyhow::Result<usize> {
    close_stale_task_prompts(tasks, documents, |_| true, result, now).await
}
