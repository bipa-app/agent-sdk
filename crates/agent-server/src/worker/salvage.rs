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
//!
//! Either way only a request that was cut off: its attempt is still open,
//! or closed `Cancelled` (the abandoned close is one). A request that
//! failed and was retried, or that completed and suspended on tool calls,
//! closed its attempt otherwise; its text is superseded or already in the
//! transcript.

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{self, ContentBlock, Message, Role};
use anyhow::{Context, Result};

use crate::journal::committed_event::CommittedEvent;
use crate::journal::effect::{
    ABORTED_RESPONSE_NOTE, CANCELLED_RESPONSE_NOTE, EffectKey, EffectKind, EffectRecord,
    EffectState, EffectSubject, EffectWrite,
};
use crate::journal::event_repository::EventRepository;
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTaskId, TaskKind, TaskStatus};
use crate::journal::transition::Write;
use crate::journal::turn_attempt::{TurnAttemptId, TurnAttemptOutcome};
use crate::journal::turn_attempt_store::TurnAttemptStore;

/// A cut-off LLM request a turn salvages.
#[derive(Clone, Debug, PartialEq)]
pub struct SalvagedRequest {
    pub key: EffectKey,
    pub owner: AgentTaskId,
    /// The turn's own request (sent after its prompt), or an earlier
    /// root's (sent before it).
    pub own: bool,
    /// The partial response and the note; empty when no text reached the
    /// journal.
    pub messages: Vec<Message>,
}

impl SalvagedRequest {
    /// The finish the continuing turn commits with the messages.
    #[must_use]
    pub fn finish(&self) -> Write {
        Write::Effect(EffectWrite::Finish {
            key: self.key.clone(),
            owner: self.owner.clone(),
            state: EffectState::Salvaged {
                captured_output: None,
            },
        })
    }
}

/// The requests the root task `task` salvages as its turn starts on
/// `thread`, oldest first.
///
/// # Errors
/// A store read that fails.
pub async fn collect(
    task_store: &dyn AgentTaskStore,
    attempt_store: &dyn TurnAttemptStore,
    event_repo: &dyn EventRepository,
    thread: &ThreadId,
    task: &AgentTaskId,
) -> Result<Vec<SalvagedRequest>> {
    let tasks = task_store
        .list_by_thread(thread)
        .await
        .context("list the thread's tasks to salvage")?;
    let mut cut = Vec::new();
    for root in tasks.iter().filter(|row| row.kind == TaskKind::RootTurn) {
        let own = &root.id == task;
        if !own && root.status != TaskStatus::Cancelled {
            continue;
        }
        let mut requests: Vec<EffectRecord> = task_store
            .effects_for_task(&root.id)
            .await
            .context("read the requests to salvage")?
            .into_iter()
            .filter(|record| record.kind() == EffectKind::LlmRequest)
            .collect();
        requests.sort_by_key(|record| record.journal_anchor);
        for (index, record) in requests.iter().enumerate() {
            let pending = if own {
                record.state.is_started()
            } else {
                record.state == EffectState::Abandoned
            };
            if !pending || !was_cut_off(attempt_store, record).await? {
                continue;
            }
            let until = requests.get(index + 1).and_then(|next| next.journal_anchor);
            cut.push((record.clone(), own, until));
        }
    }
    if cut.is_empty() {
        return Ok(Vec::new());
    }
    cut.sort_by_key(|(record, ..)| record.journal_anchor);
    let events = event_repo
        .get_events(thread)
        .await
        .context("read the journal to salvage")?;
    Ok(cut
        .into_iter()
        .map(|(record, own, until)| SalvagedRequest {
            messages: aborted_messages(
                &events,
                &record.owner,
                record.journal_anchor,
                until,
                if own {
                    ABORTED_RESPONSE_NOTE
                } else {
                    CANCELLED_RESPONSE_NOTE
                },
            ),
            key: record.key,
            owner: record.owner,
            own,
        })
        .collect())
}

/// Count and log the requests a committed turn salvaged.
///
/// Each counts the `salvaged` transition on
/// `agent_server.effects{kind="llm_request"}` and logs one info line with
/// its key, owner and how much text it carried.
pub fn report_salvaged(requests: &[SalvagedRequest]) {
    for request in requests {
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
            effect_key = request.key.as_str(),
            task_id = request.owner.as_str(),
            own = request.own,
            salvaged_chars = chars;
            "a cut-off LLM response was salvaged into the turn that continues"
        );
    }
}

/// Whether the request's attempt shows it was cut off: still open, or
/// closed `Cancelled`.
async fn was_cut_off(attempt_store: &dyn TurnAttemptStore, record: &EffectRecord) -> Result<bool> {
    let EffectSubject::Llm {
        turn_attempt_id, ..
    } = &record.subject
    else {
        return Ok(false);
    };
    let attempt = attempt_store
        .get(&TurnAttemptId::from_string(turn_attempt_id.clone()))
        .await
        .context("read the attempt of a request to salvage")?;
    Ok(attempt.is_some_and(|attempt| {
        matches!(attempt.outcome, None | Some(TurnAttemptOutcome::Cancelled))
    }))
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
    let Some(from) = from else {
        return Vec::new();
    };
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
    if blocks.is_empty() {
        return Vec::new();
    }
    vec![
        Message {
            role: Role::Assistant,
            content: llm::Content::Blocks(blocks),
        },
        Message::user(note),
    ]
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
