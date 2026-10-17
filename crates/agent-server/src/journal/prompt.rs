//! Durable prompt rows: a human prompt that is not a parked task.
//!
//! A tool confirmation and a question already have a durable row, the task
//! they park ([`TaskState::AwaitingConfirmation`],
//! [`TaskState::AwaitingQuestion`]). A host also raises prompts no task
//! parks on: a confirmation of its own, answered outside any turn (a merge
//! approval, a runaway-subagent check), or the copy of a parked task's
//! prompt on the thread a person watches. Such a prompt is a document of
//! kind [`PROMPT_DOC_KIND`], scoped to the thread whose clients must see it
//! and keyed by the id those clients answer with. It is written only by the
//! [`Write`]s this module builds, so it commits in the same transaction as
//! the host's own state change and the frame that announces it.
//!
//! [`PendingPrompt`] is the one read type over both records: a parked task
//! or a prompt row.
//!
//! A prompt row is answered once: [`StoredPrompt::answer_write`] puts the answer
//! under the revision it read, so a second answer fails the document's
//! expectation. A task-owned row is only an index of its task's prompt:
//! the task's own decision is the answer, and the row is closed after it.
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{PendingToolCallInfo, QuestionPayload, ThreadId, ToolResult, ToolTier};
use anyhow::Context;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use super::document::{
    DocExpect, DocKind, DocScope, Document, DocumentKey, DocumentStore, DocumentWrite,
    MAX_DOCUMENT_KEY_BYTES,
};
use super::store::AgentTaskStore;
use super::task::{AgentTask, AgentTaskId, TaskStatus};
use super::task_state::TaskState;
use super::transition::Write;

/// The document kind of every prompt row.
pub const PROMPT_DOC_KIND: &str = "agent.prompt";

/// The schema version of [`PromptRecord`] as stored.
pub const PROMPT_SCHEMA_VERSION: u32 = 1;

/// One step of a prompt row's life, as `agent_server.prompts` counts it.
/// A host opens a row inside its own transition, so it counts its own
/// opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PromptStep {
    Answer,
    Close,
    Withdraw,
    /// The host's handler ran on a live answer.
    Handle,
    /// The host's handler ran on an answer found unclosed at start.
    Redrive,
}

impl PromptStep {
    /// The bounded metric label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Answer => "answer",
            Self::Close => "close",
            Self::Withdraw => "withdraw",
            Self::Handle => "handle",
            Self::Redrive => "redrive",
        }
    }
}

/// Rows read per page when a read walks every prompt row.
const PAGE: u32 = 256;

/// The [`DocKind`] of [`PROMPT_DOC_KIND`].
///
/// # Errors
/// Never in practice: the constant is a valid kind.
pub fn prompt_kind() -> anyhow::Result<DocKind> {
    DocKind::new(PROMPT_DOC_KIND)
}

/// The document a prompt row lives in.
///
/// # Errors
/// See [`prompt_kind`].
pub fn prompt_key(surface_thread: &ThreadId, id: &str) -> anyhow::Result<DocumentKey> {
    Ok(DocumentKey::new(
        DocScope::Thread(surface_thread.clone()),
        prompt_kind()?,
        id,
    ))
}

/// Who answers a prompt row's decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PromptOwner {
    /// The host: its prompt handler applies the answer. `name` is the
    /// host's own `<owner>.<name>` label for the prompt, for example
    /// `bip.merge_gate`.
    Host { name: String },
    /// A task parked on a confirmation on `thread_id`; the row shows its
    /// prompt on another thread. The task's decision is the answer.
    Task {
        task_id: AgentTaskId,
        thread_id: ThreadId,
    },
}

impl PromptOwner {
    /// The bounded metric label: `host` or `task`.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Host { .. } => "host",
            Self::Task { .. } => "task",
        }
    }
}

/// The `ToolRequiresConfirmation` frame that opens a prompt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptFrame {
    pub name: String,
    pub display_name: String,
    pub input: serde_json::Value,
    pub description: String,
}

/// The tool identity of the `ToolCallStart`/`ToolCallEnd` pair that closes
/// a prompt. It may differ from the opening frame's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCloseFrame {
    pub name: String,
    pub display_name: String,
    pub input: serde_json::Value,
}

/// A person's (or a run mode's) decision on a prompt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PromptDecision {
    Approved,
    Rejected { reason: String },
    TimedOut,
}

/// The answer a prompt row holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptAnswer {
    pub decision: PromptDecision,
    /// The request that answered; the same request answering again is a
    /// replay.
    pub request_id: String,
    /// Whether a client answered with no person deciding.
    pub automatic: bool,
    #[serde(with = "time::serde::rfc3339")]
    pub answered_at: OffsetDateTime,
}

/// Where a prompt row is in its life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum PromptState {
    /// Waiting for a decision.
    Open,
    /// Decided, not yet closed: the owner has still to apply the answer.
    Answered { answer: PromptAnswer },
    /// Closed and kept, so the same id cannot open again.
    Closed {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        answer: Option<PromptAnswer>,
        /// Closed without an answer: its task resolved another way.
        withdrawn: bool,
        #[serde(with = "time::serde::rfc3339")]
        closed_at: OffsetDateTime,
    },
}

impl PromptState {
    /// The answer, when the prompt has one.
    #[must_use]
    pub const fn answer(&self) -> Option<&PromptAnswer> {
        match self {
            Self::Open => None,
            Self::Answered { answer } => Some(answer),
            Self::Closed { answer, .. } => answer.as_ref(),
        }
    }
}

/// One prompt row, as stored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptRecord {
    /// The id clients see as the frame's `tool_call_id`, and the document
    /// key.
    pub id: String,
    /// The thread whose clients see the prompt; the document scope.
    pub surface_thread: ThreadId,
    pub owner: PromptOwner,
    /// A task that waits for this host prompt, so the host's close can wake
    /// it. A task-owned row names its task in [`PromptOwner::Task`]
    /// instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_task: Option<AgentTaskId>,
    pub frame: PromptFrame,
    pub close: PromptCloseFrame,
    /// Whether only a person may approve it; the host's decision guard
    /// enforces this, the row only records it.
    #[serde(default)]
    pub must_reach_human: bool,
    /// The host's own state for applying the answer. agent-sdk never reads
    /// it.
    #[serde(default)]
    pub host_state: serde_json::Value,
    pub state: PromptState,
    #[serde(with = "time::serde::rfc3339")]
    pub opened_at: OffsetDateTime,
}

/// The parts of a new prompt row.
#[derive(Clone, Debug)]
pub struct NewPrompt {
    pub id: String,
    pub surface_thread: ThreadId,
    pub owner: PromptOwner,
    pub owner_task: Option<AgentTaskId>,
    pub frame: PromptFrame,
    pub close: PromptCloseFrame,
    pub must_reach_human: bool,
    pub host_state: serde_json::Value,
}

impl PromptRecord {
    /// An open prompt row.
    #[must_use]
    pub fn open(prompt: NewPrompt, now: OffsetDateTime) -> Self {
        Self {
            id: prompt.id,
            surface_thread: prompt.surface_thread,
            owner: prompt.owner,
            owner_task: prompt.owner_task,
            frame: prompt.frame,
            close: prompt.close,
            must_reach_human: prompt.must_reach_human,
            host_state: prompt.host_state,
            state: PromptState::Open,
            opened_at: now,
        }
    }

    /// The document key.
    ///
    /// # Errors
    /// See [`prompt_kind`].
    pub fn key(&self) -> anyhow::Result<DocumentKey> {
        prompt_key(&self.surface_thread, &self.id)
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.id.is_empty(), "a prompt row needs an id");
        anyhow::ensure!(
            self.id.len() <= MAX_DOCUMENT_KEY_BYTES,
            "prompt id is longer than {MAX_DOCUMENT_KEY_BYTES} bytes"
        );
        anyhow::ensure!(
            !self.surface_thread.0.is_empty(),
            "prompt {} names no surface thread",
            self.id
        );
        match &self.owner {
            PromptOwner::Host { name } => {
                DocKind::new(name.clone())
                    .with_context(|| format!("prompt {} owner name", self.id))?;
            }
            PromptOwner::Task { .. } => anyhow::ensure!(
                self.owner_task.is_none(),
                "task-owned prompt {} names its task in its owner",
                self.id
            ),
        }
        Ok(())
    }

    /// The writes that store this row and, when `announce`, append its
    /// `ToolRequiresConfirmation` frame to the surface thread. The row must
    /// not exist yet: opening an id that is live (open, answered, or a kept
    /// tombstone) fails the document's expectation and rejects the whole
    /// transition. A row that starts answered (`announce` off) is an answer
    /// the host recorded itself, for its handler to apply after a restart.
    ///
    /// # Errors
    /// An empty id or surface thread, an id over the key limit, an owner
    /// name that is not `<owner>.<name>`, or a closed state.
    pub fn open_writes(&self, announce: bool) -> anyhow::Result<Vec<Write>> {
        self.validate()?;
        anyhow::ensure!(
            !matches!(self.state, PromptState::Closed { .. }),
            "prompt {} cannot open closed",
            self.id
        );
        let mut writes = vec![Write::Document(DocumentWrite::put(
            self.key()?,
            PROMPT_SCHEMA_VERSION,
            DocExpect::Absent,
            serde_json::to_value(self).context("serialize prompt row")?,
        ))];
        if announce {
            writes.push(Write::AppendEvents {
                thread: self.surface_thread.clone(),
                events: vec![self.open_event()],
            });
        }
        Ok(writes)
    }

    /// The frame that announces the prompt.
    #[must_use]
    pub fn open_event(&self) -> AgentEvent {
        AgentEvent::tool_requires_confirmation(
            self.id.clone(),
            self.frame.name.clone(),
            self.frame.display_name.clone(),
            self.frame.input.clone(),
            self.frame.description.clone(),
        )
    }

    /// The pair that closes the prompt with `result`.
    #[must_use]
    pub fn close_events(&self, result: ToolResult) -> Vec<AgentEvent> {
        vec![
            AgentEvent::tool_call_start(
                self.id.clone(),
                self.close.name.clone(),
                self.close.display_name.clone(),
                self.close.input.clone(),
                ToolTier::Confirm,
            ),
            AgentEvent::tool_call_end(
                self.id.clone(),
                self.close.name.clone(),
                self.close.display_name.clone(),
                result,
            ),
        ]
    }
}

/// Why a prompt row refused a step.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PromptRefusal {
    /// The prompt already holds an answer. `same_request` is set when the
    /// stored answer came from the request now asking, which makes the ask
    /// a replay.
    #[error("prompt {id} is already answered")]
    AlreadyAnswered { id: String, same_request: bool },
    /// The prompt is closed.
    #[error("prompt {id} is closed")]
    Closed { id: String },
}

/// A prompt row with the revision it was read at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredPrompt {
    pub record: PromptRecord,
    pub revision: u64,
}

impl StoredPrompt {
    /// Decode a document of [`PROMPT_DOC_KIND`].
    ///
    /// # Errors
    /// Another kind, an unknown schema version, or a value that is not a
    /// prompt row.
    pub fn from_document(document: &Document) -> anyhow::Result<Self> {
        anyhow::ensure!(
            document.key.kind.as_str() == PROMPT_DOC_KIND,
            "document {} is not a prompt row",
            document.key
        );
        anyhow::ensure!(
            document.schema_version == PROMPT_SCHEMA_VERSION,
            "prompt row {} has schema version {}",
            document.key,
            document.schema_version
        );
        let record: PromptRecord = serde_json::from_value(document.value.clone())
            .with_context(|| format!("decode prompt row {}", document.key))?;
        Ok(Self {
            record,
            revision: document.revision,
        })
    }

    fn put(&self, record: &PromptRecord) -> anyhow::Result<Write> {
        Ok(Write::Document(DocumentWrite::put(
            record.key()?,
            PROMPT_SCHEMA_VERSION,
            DocExpect::Revision(self.revision),
            serde_json::to_value(record).context("serialize prompt row")?,
        )))
    }

    /// The write that stores `answer` on an open prompt, under the read
    /// revision: of two racing answers, one commits and the other fails
    /// the expectation.
    ///
    /// # Errors
    /// [`PromptRefusal`] (inside the `anyhow::Error`) when the prompt is
    /// not open.
    pub fn answer_write(&self, answer: PromptAnswer) -> anyhow::Result<Write> {
        match &self.record.state {
            PromptState::Open => {}
            PromptState::Answered { answer: stored }
            | PromptState::Closed {
                answer: Some(stored),
                ..
            } => {
                return Err(PromptRefusal::AlreadyAnswered {
                    id: self.record.id.clone(),
                    same_request: stored.request_id == answer.request_id,
                }
                .into());
            }
            PromptState::Closed { answer: None, .. } => {
                return Err(PromptRefusal::Closed {
                    id: self.record.id.clone(),
                }
                .into());
            }
        }
        let mut record = self.record.clone();
        record.state = PromptState::Answered { answer };
        self.put(&record)
    }

    /// The writes that close the prompt with `result`: the row becomes a
    /// kept tombstone when `retain`, else it is retired, and the closing
    /// pair is appended to the surface thread. An open prompt closes as
    /// withdrawn.
    ///
    /// # Errors
    /// [`PromptRefusal::Closed`] when the prompt is already closed.
    pub fn close_writes(
        &self,
        result: ToolResult,
        retain: bool,
        now: OffsetDateTime,
    ) -> anyhow::Result<Vec<Write>> {
        let answer = match &self.record.state {
            PromptState::Open => None,
            PromptState::Answered { answer } => Some(answer.clone()),
            PromptState::Closed { .. } => {
                return Err(PromptRefusal::Closed {
                    id: self.record.id.clone(),
                }
                .into());
            }
        };
        let row = if retain {
            let mut record = self.record.clone();
            record.state = PromptState::Closed {
                withdrawn: answer.is_none(),
                answer,
                closed_at: now,
            };
            self.put(&record)?
        } else {
            Write::Document(DocumentWrite::retire(
                self.record.key()?,
                DocExpect::Revision(self.revision),
            ))
        };
        Ok(vec![
            row,
            Write::AppendEvents {
                thread: self.record.surface_thread.clone(),
                events: self.record.close_events(result),
            },
        ])
    }

    /// The write that retires a kept tombstone, so its id may open again.
    ///
    /// # Errors
    /// See [`prompt_kind`].
    pub fn retire_write(&self) -> anyhow::Result<Write> {
        Ok(Write::Document(DocumentWrite::retire(
            self.record.key()?,
            DocExpect::Revision(self.revision),
        )))
    }
}

/// The prompt row `id` on `surface_thread`, open, answered or kept.
///
/// # Errors
/// The store failed, or the row does not decode.
pub async fn get_prompt(
    documents: &dyn DocumentStore,
    surface_thread: &ThreadId,
    id: &str,
) -> anyhow::Result<Option<StoredPrompt>> {
    documents
        .get_document(&prompt_key(surface_thread, id)?)
        .await?
        .as_ref()
        .map(StoredPrompt::from_document)
        .transpose()
}

/// Every live prompt row on `surface_thread`, ordered by id.
///
/// # Errors
/// The store failed, or a row does not decode.
pub async fn thread_prompts(
    documents: &dyn DocumentStore,
    surface_thread: &ThreadId,
) -> anyhow::Result<Vec<StoredPrompt>> {
    documents
        .list_documents(&DocScope::Thread(surface_thread.clone()), &prompt_kind()?)
        .await?
        .iter()
        .map(StoredPrompt::from_document)
        .collect()
}

/// Every live prompt row on every thread that `keep` selects, read page
/// by page.
///
/// # Errors
/// The store failed, or a row does not decode.
pub async fn all_prompts(
    documents: &dyn DocumentStore,
    keep: impl Fn(&PromptRecord) -> bool + Send,
) -> anyhow::Result<Vec<StoredPrompt>> {
    let kind = prompt_kind()?;
    let mut after: Option<DocumentKey> = None;
    let mut selected = Vec::new();
    loop {
        let page = documents
            .list_documents_by_kind(&kind, after.as_ref(), PAGE)
            .await?;
        let full = u32::try_from(page.len()).is_ok_and(|len| len == PAGE);
        for document in &page {
            let prompt = StoredPrompt::from_document(document)?;
            if keep(&prompt.record) {
                selected.push(prompt);
            }
        }
        match page.last() {
            Some(last) if full => after = Some(last.key.clone()),
            _ => return Ok(selected),
        }
    }
}

/// A human prompt waiting for a decision, whichever row holds it.
#[derive(Clone, Debug)]
pub enum PendingPrompt {
    /// A tool child parked on its confirmation.
    ToolConfirmation {
        task_id: AgentTaskId,
        thread_id: ThreadId,
        call: PendingToolCallInfo,
    },
    /// A root parked on a question batch.
    Question {
        task_id: AgentTaskId,
        thread_id: ThreadId,
        questions: Vec<QuestionPayload>,
    },
    /// An open prompt row.
    Row(StoredPrompt),
}

/// The call a parked tool child awaits.
fn awaited_call(task: &AgentTask) -> Option<&PendingToolCallInfo> {
    let continuation = task.state.continuation()?;
    let index = match task.spawn_index {
        Some(index) => usize::try_from(index).ok()?,
        None => continuation.payload.awaiting_index,
    };
    continuation.payload.pending_tool_calls.get(index)
}

/// Whether a task-owned row's task still awaits its confirmation. A row
/// whose task resolved is no longer pending, even before its close lands.
///
/// # Errors
/// The store failed.
pub async fn owner_task_awaits(
    tasks: &dyn AgentTaskStore,
    record: &PromptRecord,
) -> anyhow::Result<bool> {
    match &record.owner {
        PromptOwner::Host { .. } => Ok(true),
        PromptOwner::Task { task_id, .. } => Ok(tasks
            .get(task_id)
            .await?
            .is_some_and(|task| task.status == TaskStatus::AwaitingConfirmation)),
    }
}

/// Every prompt on `thread` that waits for a decision: its parked tool
/// confirmations and questions, then its open prompt rows (a task-owned row
/// only while its task still awaits).
///
/// # Errors
/// A store failed, or a row does not decode.
pub async fn pending_prompts(
    tasks: &dyn AgentTaskStore,
    documents: &dyn DocumentStore,
    thread: &ThreadId,
) -> anyhow::Result<Vec<PendingPrompt>> {
    let mut pending = Vec::new();
    for task in tasks.list_by_thread(thread).await? {
        match &task.state {
            TaskState::AwaitingConfirmation { .. } => {
                if let Some(call) = awaited_call(&task) {
                    pending.push(PendingPrompt::ToolConfirmation {
                        task_id: task.id.clone(),
                        thread_id: task.thread_id.clone(),
                        call: call.clone(),
                    });
                }
            }
            TaskState::AwaitingQuestion { questions, .. } => {
                pending.push(PendingPrompt::Question {
                    task_id: task.id.clone(),
                    thread_id: task.thread_id.clone(),
                    questions: questions.clone(),
                });
            }
            TaskState::None
            | TaskState::WaitingOnChildren { .. }
            | TaskState::AnsweredQuestion { .. }
            | TaskState::SubagentInvocation { .. }
            | TaskState::ReadyToResume { .. } => {}
        }
    }
    for prompt in thread_prompts(documents, thread).await? {
        if matches!(prompt.record.state, PromptState::Open)
            && owner_task_awaits(tasks, &prompt.record).await?
        {
            pending.push(PendingPrompt::Row(prompt));
        }
    }
    Ok(pending)
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = anyhow::Result<()>;

    fn now() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH
    }

    fn host_prompt(id: &str) -> PromptRecord {
        PromptRecord::open(
            NewPrompt {
                id: id.to_owned(),
                surface_thread: ThreadId::from_string("session"),
                owner: PromptOwner::Host {
                    name: "host.merge".to_owned(),
                },
                frame: PromptFrame {
                    name: "apply_merge".to_owned(),
                    display_name: "Merge".to_owned(),
                    input: serde_json::json!({"branch": "b"}),
                    description: "merge b?".to_owned(),
                },
                close: PromptCloseFrame {
                    name: "apply_merge".to_owned(),
                    display_name: "Merge".to_owned(),
                    input: serde_json::json!({"branch": "b"}),
                },
                owner_task: None,
                must_reach_human: false,
                host_state: serde_json::json!({"commit": "abc"}),
            },
            now(),
        )
    }

    fn stored(record: PromptRecord, revision: u64) -> StoredPrompt {
        StoredPrompt { record, revision }
    }

    fn answer(request: &str) -> PromptAnswer {
        PromptAnswer {
            decision: PromptDecision::Rejected {
                reason: "no".to_owned(),
            },
            request_id: request.to_owned(),
            automatic: false,
            answered_at: now(),
        }
    }

    fn document_of(write: &Write) -> anyhow::Result<&DocumentWrite> {
        match write {
            Write::Document(document) => Ok(document),
            other => anyhow::bail!("expected a document write, got {other:?}"),
        }
    }

    #[test]
    fn open_puts_an_absent_row_and_announces_it_once() -> TestResult {
        let record = host_prompt("merge-1");
        let writes = record.open_writes(true)?;
        assert_eq!(writes.len(), 2);
        let row = document_of(&writes[0])?;
        assert_eq!(row.expect, DocExpect::Absent);
        assert_eq!(
            row.key,
            prompt_key(&ThreadId::from_string("session"), "merge-1")?
        );
        let Write::AppendEvents { thread, events } = &writes[1] else {
            anyhow::bail!("expected the open frame");
        };
        assert_eq!(thread.0, "session");
        assert!(matches!(
            events.as_slice(),
            [AgentEvent::ToolRequiresConfirmation { id, name, .. }]
                if id == "merge-1" && name == "apply_merge"
        ));
        assert_eq!(
            record.open_writes(false)?.len(),
            1,
            "an unannounced open writes only the row"
        );
        Ok(())
    }

    #[test]
    fn open_refuses_a_malformed_row() {
        let mut no_id = host_prompt("");
        assert!(no_id.open_writes(true).is_err());
        no_id.id = "x".repeat(MAX_DOCUMENT_KEY_BYTES + 1);
        assert!(no_id.open_writes(true).is_err());
        let mut bad_owner = host_prompt("p");
        bad_owner.owner = PromptOwner::Host {
            name: "merge".to_owned(),
        };
        assert!(bad_owner.open_writes(true).is_err());
        let mut double_task = host_prompt("p");
        double_task.owner = PromptOwner::Task {
            task_id: AgentTaskId::from_string("t"),
            thread_id: ThreadId::from_string("child"),
        };
        double_task.owner_task = Some(AgentTaskId::from_string("t"));
        assert!(double_task.open_writes(true).is_err());
    }

    #[test]
    fn answer_puts_under_the_read_revision_and_refuses_a_second() -> TestResult {
        let open = stored(host_prompt("p"), 3);
        let write = open.answer_write(answer("r1"))?;
        let row = document_of(&write)?;
        assert_eq!(row.expect, DocExpect::Revision(3));

        let mut answered = open.record;
        answered.state = PromptState::Answered {
            answer: answer("r1"),
        };
        let answered = stored(answered, 4);
        let replay = answered.answer_write(answer("r1"));
        assert_eq!(
            replay
                .err()
                .and_then(|error| error.downcast::<PromptRefusal>().ok()),
            Some(PromptRefusal::AlreadyAnswered {
                id: "p".to_owned(),
                same_request: true
            })
        );
        let conflict = answered.answer_write(answer("r2"));
        assert_eq!(
            conflict
                .err()
                .and_then(|error| error.downcast::<PromptRefusal>().ok()),
            Some(PromptRefusal::AlreadyAnswered {
                id: "p".to_owned(),
                same_request: false
            })
        );
        Ok(())
    }

    #[test]
    fn close_retires_or_keeps_a_tombstone_and_appends_one_pair() -> TestResult {
        let mut answered = host_prompt("p");
        answered.state = PromptState::Answered {
            answer: answer("r1"),
        };
        let answered = stored(answered, 4);

        let retired = answered.close_writes(ToolResult::success("merged"), false, now())?;
        assert_eq!(document_of(&retired[0])?.op, super::super::DocOp::Retire);
        assert_eq!(document_of(&retired[0])?.expect, DocExpect::Revision(4));
        let Write::AppendEvents { events, .. } = &retired[1] else {
            anyhow::bail!("expected the close pair");
        };
        assert!(matches!(
            events.as_slice(),
            [
                AgentEvent::ToolCallStart { id: start, tier: ToolTier::Confirm, .. },
                AgentEvent::ToolCallEnd { id: end, result, .. },
            ] if start == "p" && end == "p" && result.success
        ));

        let kept = answered.close_writes(ToolResult::success("merged"), true, now())?;
        let super::super::DocOp::Put(value) = &document_of(&kept[0])?.op else {
            anyhow::bail!("a kept close puts the tombstone");
        };
        let tombstone: PromptRecord = serde_json::from_value(value.clone())?;
        assert!(matches!(
            tombstone.state,
            PromptState::Closed {
                withdrawn: false,
                answer: Some(_),
                ..
            }
        ));
        let closed = stored(tombstone, 5);
        assert!(
            closed
                .close_writes(ToolResult::success("again"), true, now())
                .is_err()
        );
        assert!(closed.answer_write(answer("r1")).is_err());
        Ok(())
    }

    #[test]
    fn closing_an_open_prompt_withdraws_it() -> TestResult {
        let open = stored(host_prompt("p"), 1);
        let kept = open.close_writes(ToolResult::error("withdrawn"), true, now())?;
        let super::super::DocOp::Put(value) = &document_of(&kept[0])?.op else {
            anyhow::bail!("a kept close puts the tombstone");
        };
        let tombstone: PromptRecord = serde_json::from_value(value.clone())?;
        assert!(matches!(
            tombstone.state,
            PromptState::Closed {
                withdrawn: true,
                answer: None,
                ..
            }
        ));
        let closed = stored(tombstone, 2);
        assert!(matches!(
            closed
                .answer_write(answer("r1"))
                .err()
                .and_then(|error| error.downcast::<PromptRefusal>().ok()),
            Some(PromptRefusal::Closed { .. })
        ));
        Ok(())
    }

    #[test]
    fn rows_round_trip_through_their_document() -> TestResult {
        let mut record = host_prompt("p");
        record.owner = PromptOwner::Task {
            task_id: AgentTaskId::from_string("task-1"),
            thread_id: ThreadId::from_string("child"),
        };
        let document = Document {
            key: record.key()?,
            schema_version: PROMPT_SCHEMA_VERSION,
            revision: 7,
            value: serde_json::to_value(&record)?,
            updated_at: now(),
        };
        assert_eq!(StoredPrompt::from_document(&document)?, stored(record, 7));
        let newer = Document {
            schema_version: PROMPT_SCHEMA_VERSION + 1,
            ..document
        };
        assert!(StoredPrompt::from_document(&newer).is_err());
        Ok(())
    }
}
