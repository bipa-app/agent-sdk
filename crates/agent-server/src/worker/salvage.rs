//! Salvage of LLM requests cut off before they finished (crash safety
//! Phase 2, P2-3c).
//!
//! An LLM request is an effect ([`crate::journal::effect`]) started with
//! its attempt row, and its streamed text reaches the journal as
//! `TextDelta` events while it runs. A request cut off by a crash or a
//! cancel never commits a turn, so the next request would lose that text.
//! Salvage turns it back into messages: the partial response as an
//! `assistant` message, then a note as a `user` message:
//! [`ABORTED_RESPONSE_NOTE`] ("continue from where it ends") after a crash,
//! [`CANCELLED_RESPONSE_NOTE`] ("the user stopped it") after a user's stop. The turn that continues sends them with its request and
//! commits them with its turn, in the same transaction as the request's
//! `Finish(Salvaged)`. Until that commit the request stays started (or
//! abandoned) and its text stays in the journal, so a turn that dies again
//! salvages it again, never twice into one transcript.
//!
//! Which requests are salvaged:
//! - **The turn's own**: a re-acquired root task's `Started` requests (the
//!   process died mid-stream, or its lease expired).
//! - **A cancelled root's**: an earlier root on the thread that ended
//!   `Cancelled` with an `Abandoned` request (a cancel mid-stream, a crash
//!   during the cancel).
//! - **A failed root's**: an earlier root that ended `Failed` since the
//!   thread's last commit, with an `Abandoned` request. However it failed (a
//!   provider error, a worker error, a fail-close when its attempts ran
//!   out at acquisition, in the lease sweep or in an epoch reclaim), the
//!   terminal transition abandoned its started requests.
//!
//! A failed root also leaves a note, whether or not a response was
//! streaming: [`FailedTurnClass::note`] says the turn failed and why, after
//! its cut-off response if it had one. The tool calls it completed are
//! already in the next turn's context, from the draft its suspensions left
//! (see [`crate::journal::thread_recover`]), and commit with that turn. The
//! note has no request to finish, so it is told apart by the commit itself:
//! only roots from the latest checkpoint's owner on get one (every root, on
//! a fork that has not committed yet: its checkpoint came from its source),
//! and the turn that adds it commits a new checkpoint. A turn that suspends carries the
//! note in its draft until it commits, so roots whose note the draft already
//! holds are skipped.
//!
//! Either way only a request that was cut off: its attempt is still open,
//! or closed `Cancelled` (the abandoned close is one). A request that
//! failed and was retried, or that completed and suspended on tool calls,
//! closed its attempt otherwise; its text is superseded or already in the
//! transcript. The one exception is a failed root's last request when the
//! provider failed it: nothing retried it, so its streamed text is the last
//! the turn produced.

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{self, ContentBlock, Message, Role};
use anyhow::{Context, Result};

use crate::journal::committed_event::CommittedEvent;
use crate::journal::effect::{
    ABORTED_RESPONSE_NOTE, CANCELLED_RESPONSE_NOTE, EffectKey, EffectKind, EffectRecord,
    EffectState, EffectSubject, EffectWrite, FailedTurnClass, is_failed_turn_note_text,
};
use crate::journal::event_repository::EventRepository;
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTaskId, TaskKind, TaskStatus};
use crate::journal::transition::Write;
use crate::journal::turn_attempt::{TurnAttemptId, TurnAttemptOutcome};
use crate::journal::turn_attempt_store::TurnAttemptStore;

/// A cut-off LLM request a turn salvages, or a failed root's note.
#[derive(Clone, Debug, PartialEq)]
pub struct SalvagedRequest {
    /// The request to finish `Salvaged`; `None` for a failed root's note, and
    /// for a failed root's last request that finished `Failed`, which nothing
    /// can finish again.
    pub key: Option<EffectKey>,
    pub owner: AgentTaskId,
    /// The turn's own request (sent after its prompt), or an earlier
    /// root's (sent before it).
    pub own: bool,
    /// The partial response and the note; empty when no text reached the
    /// journal.
    pub messages: Vec<Message>,
}

impl SalvagedRequest {
    /// The finish the continuing turn commits with the messages; none for a
    /// failed root's note.
    #[must_use]
    pub fn finish(&self) -> Option<Write> {
        Some(Write::Effect(EffectWrite::Finish {
            key: self.key.clone()?,
            owner: self.owner.clone(),
            state: EffectState::Salvaged {
                captured_output: None,
            },
        }))
    }
}

/// Where the thread stands for salvage as a turn starts.
#[derive(Clone, Copy, Debug, Default)]
pub struct SalvageScope<'a> {
    /// The owner of the thread's latest checkpoint; `None` on a thread with
    /// no commit yet. A failed root from it on is unreported; on a fork,
    /// whose checkpoint came from its source, every failed root is.
    pub committed_by: Option<&'a AgentTaskId>,
    /// The failed-turn notes the draft already holds, from a turn that
    /// added them and suspended. That many of the oldest unreported failed
    /// roots are skipped.
    pub noted: usize,
}

impl<'a> SalvageScope<'a> {
    /// The scope of a turn whose context is the latest checkpoint
    /// `committed_by` produced, followed by `draft`.
    #[must_use]
    pub fn new(committed_by: Option<&'a AgentTaskId>, draft: &[Message]) -> Self {
        Self {
            committed_by,
            noted: draft
                .iter()
                .filter(|message| holds_failed_turn_note(message))
                .count(),
        }
    }
}

fn holds_failed_turn_note(message: &Message) -> bool {
    message.role == Role::User && match &message.content {
        llm::Content::Text(text) => is_failed_turn_note_text(text),
        llm::Content::Blocks(blocks) => blocks.iter().any(
            |block| matches!(block, ContentBlock::Text { text } if is_failed_turn_note_text(text)),
        ),
    }
}

/// What the root task `task` salvages as its turn starts on `thread`, in
/// order: each earlier root's cut-off requests and, for a failed root, its
/// note, then the turn's own requests.
///
/// # Errors
/// A store read that fails.
pub async fn collect(
    task_store: &dyn AgentTaskStore,
    attempt_store: &dyn TurnAttemptStore,
    event_repo: &dyn EventRepository,
    thread: &ThreadId,
    task: &AgentTaskId,
    scope: SalvageScope<'_>,
) -> Result<Vec<SalvagedRequest>> {
    let tasks = task_store
        .list_by_thread(thread)
        .await
        .context("list the thread's tasks to salvage")?;
    let mut cut = Vec::new();
    // A checkpoint no root of this thread produced was inherited: a fork
    // copies its source's latest checkpoint, owner included. Every root on
    // the thread then came after it, as on a thread with no commit.
    let mut since_commit = scope.committed_by.is_none_or(|owner| {
        !tasks
            .iter()
            .any(|row| row.kind == TaskKind::RootTurn && &row.id == owner)
    });
    let mut unreported = 0_usize;
    for root in tasks.iter().filter(|row| row.kind == TaskKind::RootTurn) {
        since_commit |= scope.committed_by == Some(&root.id);
        let own = &root.id == task;
        let failed = !own && since_commit && root.status == TaskStatus::Failed;
        if failed {
            unreported += 1;
            if unreported <= scope.noted {
                continue;
            }
        }
        let cause = if own {
            Cut::Crash
        } else if failed {
            Cut::Failure
        } else if root.status == TaskStatus::Cancelled {
            Cut::Stop
        } else {
            continue;
        };
        cut.extend(cut_requests(task_store, attempt_store, root, cause).await?);
        if failed {
            cut.push(Salvage::FailedNote {
                owner: root.id.clone(),
                class: FailedTurnClass::of(root),
            });
        }
    }
    if cut.is_empty() {
        return Ok(Vec::new());
    }
    let events = if cut
        .iter()
        .any(|salvage| matches!(salvage, Salvage::Request { .. }))
    {
        event_repo
            .get_events(thread)
            .await
            .context("read the journal to salvage")?
    } else {
        Vec::new()
    };
    Ok(cut
        .into_iter()
        .map(|salvage| salvage.into_request(&events))
        .collect())
}

/// `root`'s cut-off requests, oldest first: its `Started` ones when it is
/// the turn's own root, its `Abandoned` ones otherwise. A failed root also
/// gives its last request when the provider failed it (its attempt closed
/// with a provider error): that response was the turn's last, and nothing
/// retried it. Its request is `Abandoned` (the root's end abandoned it) or
/// `Failed` (it finished with the error); a `Failed` one cannot become
/// `Salvaged`, so it carries no finish and, like the note, is kept to one
/// report by the commit.
async fn cut_requests(
    task_store: &dyn AgentTaskStore,
    attempt_store: &dyn TurnAttemptStore,
    root: &crate::journal::task::AgentTask,
    cause: Cut,
) -> Result<Vec<Salvage>> {
    let mut requests: Vec<EffectRecord> = task_store
        .effects_for_task(&root.id)
        .await
        .context("read the requests to salvage")?
        .into_iter()
        .filter(|record| record.kind() == EffectKind::LlmRequest)
        .collect();
    requests.sort_by_key(|record| record.journal_anchor);
    let mut cut = Vec::new();
    let last = requests.len().checked_sub(1);
    for (index, record) in requests.iter().enumerate() {
        let finish = match cause {
            Cut::Crash => record.state.is_started(),
            Cut::Stop | Cut::Failure => record.state == EffectState::Abandoned,
        };
        let failed_last = cause == Cut::Failure
            && Some(index) == last
            && (finish || matches!(record.state, EffectState::Failed { .. }));
        let salvaged = match attempt_outcome(attempt_store, record).await? {
            Some(None | Some(TurnAttemptOutcome::Cancelled)) => finish,
            Some(Some(
                TurnAttemptOutcome::ServerError
                | TurnAttemptOutcome::RateLimited
                | TurnAttemptOutcome::InvalidRequest,
            )) => failed_last,
            Some(Some(TurnAttemptOutcome::Success)) | None => false,
        };
        if !salvaged {
            continue;
        }
        cut.push(Salvage::Request {
            record: Box::new(record.clone()),
            until: requests.get(index + 1).and_then(|next| next.journal_anchor),
            cause,
            finish,
        });
    }
    Ok(cut)
}

/// Why a salvaged request was cut off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cut {
    /// The turn's own request: its process stopped mid-stream.
    Crash,
    /// A cancelled root's: the user stopped it.
    Stop,
    /// A failed root's: its turn failed.
    Failure,
}

enum Salvage {
    Request {
        record: Box<EffectRecord>,
        until: Option<u64>,
        cause: Cut,
        /// The request can be finished `Salvaged` (it is `Started` or
        /// `Abandoned`); a `Failed` one cannot.
        finish: bool,
    },
    FailedNote {
        owner: AgentTaskId,
        class: FailedTurnClass,
    },
}

impl Salvage {
    /// What the turn sends and commits for it, the request's text read from
    /// `events`.
    fn into_request(self, events: &[CommittedEvent]) -> SalvagedRequest {
        match self {
            Self::Request {
                record,
                until,
                cause,
                finish,
            } => {
                let response =
                    aborted_response(events, &record.owner, record.journal_anchor, until);
                let note = match cause {
                    Cut::Crash => Some(ABORTED_RESPONSE_NOTE),
                    Cut::Stop => Some(CANCELLED_RESPONSE_NOTE),
                    // The failed root's note follows its last request.
                    Cut::Failure => None,
                };
                let messages = response.map_or_else(Vec::new, |response| {
                    std::iter::once(response)
                        .chain(note.map(Message::user))
                        .collect()
                });
                SalvagedRequest {
                    messages,
                    key: finish.then_some(record.key),
                    owner: record.owner,
                    own: cause == Cut::Crash,
                }
            }
            Self::FailedNote { owner, class } => SalvagedRequest {
                key: None,
                owner,
                own: false,
                messages: vec![Message::user(class.note())],
            },
        }
    }
}

/// Count and log the requests a committed turn salvaged.
///
/// Each counts the `salvaged` transition on
/// `agent_server.effects{kind="llm_request"}` and logs one info line with
/// its key, owner and how much text it carried.
pub fn report_salvaged(requests: &[SalvagedRequest]) {
    for request in requests {
        let Some(key) = &request.key else {
            log::info!(
                task_id = request.owner.as_str();
                "a failed root turn's note or response was committed with the turn that continues"
            );
            continue;
        };
        crate::journal::effect::record_effect(EffectKind::LlmRequest, "salvaged");
        let chars: usize = request
            .messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .map(|message| match &message.content {
                llm::Content::Text(text) => text.len(),
                llm::Content::Blocks(blocks) => blocks
                    .iter()
                    .map(|block| match block {
                        ContentBlock::Text { text } => text.len(),
                        _ => 0,
                    })
                    .sum(),
            })
            .sum();
        log::info!(
            effect_key = key.as_str(),
            task_id = request.owner.as_str(),
            own = request.own,
            salvaged_chars = chars;
            "a cut-off LLM response was salvaged into the turn that continues"
        );
    }
}

/// How the request's attempt closed: `Some(None)` while it is still open,
/// `None` when the request has no attempt to read. A request is cut off
/// when its attempt is open or closed `Cancelled`.
async fn attempt_outcome(
    attempt_store: &dyn TurnAttemptStore,
    record: &EffectRecord,
) -> Result<Option<Option<TurnAttemptOutcome>>> {
    let EffectSubject::Llm {
        turn_attempt_id, ..
    } = &record.subject
    else {
        return Ok(None);
    };
    let attempt = attempt_store
        .get(&TurnAttemptId::from_string(turn_attempt_id.clone()))
        .await
        .context("read the attempt of a request to salvage")?;
    Ok(attempt.map(|attempt| attempt.outcome))
}

/// The messages a cut-off request becomes: its streamed text, then `note`
/// ([`ABORTED_RESPONSE_NOTE`] after a crash, [`CANCELLED_RESPONSE_NOTE`]
/// after a user's stop).
///
/// The text is the `TextDelta`s `owner` emitted from the request's journal
/// anchor up to the next request's, one text block per `message_id`.
/// Thinking is dropped (its signature is never journaled, and providers
/// refuse unsigned thinking) and a partial tool call is never journaled.
/// No text, no messages.
#[must_use]
pub fn aborted_messages(
    events: &[CommittedEvent],
    owner: &AgentTaskId,
    from: Option<u64>,
    until: Option<u64>,
    note: &str,
) -> Vec<Message> {
    aborted_response(events, owner, from, until)
        .map_or_else(Vec::new, |response| vec![response, Message::user(note)])
}

/// The `assistant` message a cut-off request's streamed text becomes; see
/// [`aborted_messages`]. `None` when no text reached the journal.
fn aborted_response(
    events: &[CommittedEvent],
    owner: &AgentTaskId,
    from: Option<u64>,
    until: Option<u64>,
) -> Option<Message> {
    let from = from?;
    let mut blocks: Vec<(String, String)> = Vec::new();
    for committed in events {
        if committed.sequence < from || until.is_some_and(|until| committed.sequence >= until) {
            continue;
        }
        let AgentEvent::TextDelta {
            message_id, delta, ..
        } = &committed.event
        else {
            continue;
        };
        if committed.event.emitter_task_id() != Some(owner.as_str()) {
            continue;
        }
        match blocks.iter_mut().find(|(id, _)| id == message_id) {
            Some((_, text)) => text.push_str(delta),
            None => blocks.push((message_id.clone(), delta.clone())),
        }
    }
    let blocks: Vec<ContentBlock> = blocks
        .into_iter()
        .filter(|(_, text)| !text.trim().is_empty())
        .map(|(_, text)| ContentBlock::Text { text })
        .collect();
    (!blocks.is_empty()).then_some(Message {
        role: Role::Assistant,
        content: llm::Content::Blocks(blocks),
    })
}

/// Merge each run of adjacent messages with the same role into one, its
/// content blocks in order. Salvage can put a note next to the turn's
/// prompt, or a cut-off response next to the previous answer.
#[must_use]
pub fn merge_adjacent_roles(messages: Vec<Message>) -> Vec<Message> {
    let mut merged: Vec<Message> = Vec::with_capacity(messages.len());
    for message in messages {
        match merged.last_mut() {
            Some(last) if last.role == message.role => {
                let mut blocks = blocks_of(std::mem::replace(
                    &mut last.content,
                    llm::Content::Blocks(Vec::new()),
                ));
                blocks.extend(blocks_of(message.content));
                last.content = llm::Content::Blocks(blocks);
            }
            _ => merged.push(message),
        }
    }
    merged
}

fn blocks_of(content: llm::Content) -> Vec<ContentBlock> {
    match content {
        llm::Content::Text(text) => vec![ContentBlock::Text { text }],
        llm::Content::Blocks(blocks) => blocks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(sequence: u64, owner: &str, message_id: &str, text: &str) -> CommittedEvent {
        CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: ThreadId::from_string("t"),
            sequence,
            timestamp: time::OffsetDateTime::UNIX_EPOCH,
            event: AgentEvent::TextDelta {
                message_id: message_id.into(),
                delta: text.into(),
                emitter_task_id: Some(owner.into()),
            },
        }
    }

    #[test]
    fn a_cut_off_response_becomes_its_text_then_the_note() {
        let owner = AgentTaskId::from_string("task_a");
        let events = vec![
            delta(3, "task_a", "m0", "before the request"),
            delta(5, "task_a", "m1", "half "),
            delta(6, "task_other", "m1", "not ours"),
            delta(7, "task_a", "m1", "an answer"),
            delta(8, "task_a", "m2", "a second block"),
            delta(9, "task_a", "m3", "the next request's"),
        ];
        let messages = aborted_messages(&events, &owner, Some(4), Some(9), ABORTED_RESPONSE_NOTE);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::Assistant);
        assert_eq!(
            messages[0].content,
            llm::Content::Blocks(vec![
                ContentBlock::Text {
                    text: "half an answer".into()
                },
                ContentBlock::Text {
                    text: "a second block".into()
                },
            ])
        );
        assert!(crate::journal::effect::is_aborted_response_note(
            &messages[1]
        ));
    }

    #[test]
    fn no_text_means_no_messages() {
        let owner = AgentTaskId::from_string("task_a");
        let none: Vec<Message> = Vec::new();
        assert_eq!(
            aborted_messages(&[], &owner, Some(0), None, ABORTED_RESPONSE_NOTE),
            none
        );
        let events = vec![delta(2, "task_a", "m1", "   ")];
        assert_eq!(
            aborted_messages(&events, &owner, Some(0), None, ABORTED_RESPONSE_NOTE),
            none
        );
        assert_eq!(
            aborted_messages(&events, &owner, None, None, ABORTED_RESPONSE_NOTE),
            none
        );
    }

    #[test]
    fn adjacent_messages_of_one_role_merge() {
        let merged = merge_adjacent_roles(vec![
            Message::user("q"),
            Message::assistant("partial"),
            Message::user(ABORTED_RESPONSE_NOTE),
            Message::user("next prompt"),
        ]);
        assert_eq!(merged.len(), 3);
        assert_eq!(
            merged[2].content,
            llm::Content::Blocks(vec![
                ContentBlock::Text {
                    text: ABORTED_RESPONSE_NOTE.into()
                },
                ContentBlock::Text {
                    text: "next prompt".into()
                },
            ])
        );
    }
}
