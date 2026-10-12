//! Durable message projection schema and invariants.
//!
//! The `message_projection` is the single place where the committed
//! per-thread conversation history lives. The committed history is
//! updated **only** at turn-completion time, so a crash can never leave
//! the committed projection ahead of the latest completed checkpoint.
//!
//! # Draft history
//!
//! Alongside the committed history, the projection carries an optional
//! `draft_messages` list — the in-flight conversation accumulated since
//! the last committed turn. Drafts exist so a turn that fails
//! mid-stream (e.g. a transient LLM transport error after several
//! suspension/resume cycles) can still surface its accumulated work to
//! the next turn instead of vanishing entirely.
//!
//! Draft semantics:
//! - Written at every tool-boundary suspension via
//!   [`MessageProjection::set_draft`]: the suspension path snapshots
//!   its full `suspended_messages` list into the draft slot.
//! - Cleared on every successful turn commit via
//!   [`MessageProjection::clear_draft`]: the committed history now
//!   subsumes the draft, so the slot must be empty for the next
//!   turn.
//! - Survives task failure: `fail_root_turn` clears the failed task's
//!   transient `TaskState` payload, but the projection's draft is
//!   preserved on a separate row, so [`super::thread_recover`] can fold
//!   the in-flight messages into the recovery view.
//!
//! # Mutation paths
//!
//! | Transition | What it does | Guard |
//! |------------|--------------|-------|
//! | [`MessageProjection::append_committed`] | Extends committed history, bumps version | Non-empty input |
//! | [`MessageProjection::replace_history`] | Atomic swap of committed history, bumps version | — |
//! | [`MessageProjection::set_draft`] | Replace the in-flight draft, bumps version | — |
//! | [`MessageProjection::clear_draft`] | Drop the in-flight draft, bumps version | — |
//!
//! All transitions consume `self` and return a new `MessageProjection`,
//! following the same pure-transition pattern as
//! [`super::thread::Thread::apply_committed_turn`].
//!
//! # Version field
//!
//! Every mutation increments `version` by one. This lets future
//! database-backed implementations use optimistic concurrency control
//! (compare-and-swap on `version`) without changing the schema layer.
//!
//! # Wire format
//!
//! Messages are stored in insertion order as `Vec<llm::Message>`.
//! The outer projection row serializes with `snake_case` keys to
//! match every other journal type. `draft_messages` defaults to an
//! empty list so older serialized rows still deserialize cleanly.

use agent_sdk_foundation::{ThreadId, llm};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Structural errors for [`MessageProjection`] transitions.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MessageProjectionError {
    #[error("cannot commit an empty message batch")]
    EmptyCommit,
    #[error(
        "compaction source count mismatch: projection exposes {available} messages, compactor saw {provided}"
    )]
    CompactionSourceMismatch { available: usize, provided: usize },
    #[error(
        "compaction retained {retained} messages, but the compacted result contains only {result_count}"
    )]
    InvalidCompactionRetention {
        retained: usize,
        result_count: usize,
    },
    #[error(
        "compaction result breaks the provider tool_use/tool_result sequence at effective message index {index}; refusing to persist it"
    )]
    CompactionBreaksToolSequence { index: usize },
}
/// Where a compaction folded a parked turn's messages into the raw
/// transcript; see [`MessageProjection::parked_fold`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParkedFold {
    /// Raw index of the first folded parked message.
    pub start: usize,
    /// How many parked messages the fold holds: all of them.
    pub len: usize,
}

/// One append-only compaction boundary in a message projection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactionEntry {
    /// First raw committed message newly covered by this compaction.
    pub compacted_start: usize,
    /// First raw committed message retained verbatim after this compaction.
    pub compacted_end: usize,
    /// Synthetic context prefix replacing the compacted raw range for LLM calls.
    pub replacement_messages: Vec<llm::Message>,
    /// True only when this entry inherited or was created from a validated
    /// generated summary. Older rows deserialize to false so typed markers
    /// cannot gain provenance merely by being loaded after an upgrade.
    #[serde(default)]
    pub generated_summary: bool,
    /// Number of messages in the effective source view seen by the compactor.
    pub source_message_count: usize,
    /// Raw messages this entry moved in from the in-flight draft:
    /// `messages[draft_fold_start..draft_fold_start + draft_fold_len]`.
    /// Rows written before these fields existed decode as an empty fold at
    /// index 0.
    #[serde(default)]
    pub draft_fold_start: usize,
    #[serde(default)]
    pub draft_fold_len: usize,
    /// Leading folded draft messages a parked turn resumed from: everything
    /// through the draft's last assistant message, since a resume stages
    /// only user messages (fan-in, injections, steering) after the parked
    /// ones. Zero when the draft parked nothing.
    #[serde(default)]
    pub draft_parked_len: usize,
    /// When this compaction entry was appended.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

// ─────────────────────────────────────────────────────────────────────
// MessageProjection
// ─────────────────────────────────────────────────────────────────────

/// One row in the `message_projection` table.
///
/// [`Self::messages`] is the append-only raw transcript.
/// [`Self::compactions`] records range-addressed effective-view prefixes;
/// [`Self::context_history`] rebuilds the LLM-facing view without rewriting
/// raw messages.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MessageProjection {
    /// The thread this projection belongs to.
    pub thread_id: ThreadId,
    /// Ordered committed message history.
    pub messages: Vec<llm::Message>,
    /// In-flight conversation accumulated since the last committed
    /// turn.
    ///
    /// Populated by [`Self::set_draft`] at every tool-boundary
    /// suspension and cleared by [`Self::clear_draft`] at turn
    /// commit. Empty when no turn is in flight.
    ///
    /// `#[serde(default)]` keeps older serialized rows compatible —
    /// they decode with an empty draft slot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub draft_messages: Vec<llm::Message>,
    /// Append-only compaction lineage. Raw committed messages remain in
    /// [`Self::messages`]; the latest entry selects the effective LLM view.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compactions: Vec<CompactionEntry>,
    /// Count of history rewrites: [`Self::replace_history`],
    /// [`Self::append_compaction`], and [`Self::append_repair`] advance it;
    /// append-only commits never do. Unlike `compactions.len()` it survives
    /// `replace_history` clearing the compaction lineage. Rows written before
    /// the field existed decode as 0.
    #[serde(default)]
    pub history_generation: u64,
    /// Monotonically increasing version, bumped on every mutation.
    pub version: u64,
    /// When this projection row was first created.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Last time this row was modified.
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

fn demote_untrusted_compaction_summary(mut message: llm::Message) -> llm::Message {
    if let llm::Content::Blocks(blocks) = &mut message.content {
        for block in blocks {
            if let llm::ContentBlock::CompactionSummary { text, .. } = block {
                *block = llm::ContentBlock::Text {
                    text: std::mem::take(text),
                };
            }
        }
    }
    message
}

fn parked_head_len(draft: &[llm::Message]) -> usize {
    draft
        .iter()
        .rposition(|message| message.role == llm::Role::Assistant)
        .map_or(0, |index| index + 1)
}

fn is_generated_compaction_summary(message: &llm::Message) -> bool {
    let is_legacy_singleton = message.role == llm::Role::User
        && matches!(
            &message.content,
            llm::Content::Blocks(blocks)
                if matches!(
                    blocks.as_slice(),
                    [llm::ContentBlock::CompactionSummary {
                        snapcompact: None,
                        ..
                    }]
                )
        );
    is_legacy_singleton || llm::canonical_snapcompact_checkpoint(message).is_some()
}

impl MessageProjection {
    /// Create a fresh, empty projection for the given thread.
    #[must_use]
    pub const fn new(thread_id: ThreadId, now: OffsetDateTime) -> Self {
        Self {
            thread_id,
            messages: Vec::new(),
            draft_messages: Vec::new(),
            compactions: Vec::new(),
            history_generation: 0,
            version: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Append committed messages to the history.
    ///
    /// This is the append-only mutation path used at turn completion.
    /// Each call extends the existing history and bumps the version.
    ///
    /// # Errors
    /// Returns [`MessageProjectionError::EmptyCommit`] if `new_messages`
    /// is empty — callers should not commit zero messages.
    pub fn append_committed(
        mut self,
        new_messages: Vec<llm::Message>,
        now: OffsetDateTime,
    ) -> Result<Self, MessageProjectionError> {
        if new_messages.is_empty() {
            return Err(MessageProjectionError::EmptyCommit);
        }
        self.messages.extend(new_messages);
        self.version += 1;
        self.updated_at = now;
        Ok(self)
    }

    /// Explicitly replace the raw history for non-compaction repair/import paths.
    ///
    /// Compaction must use [`Self::append_compaction`]. A replacement invalidates
    /// prior compaction boundaries and drops the in-flight draft atomically.
    /// Empty replacements are allowed.
    #[must_use]
    pub fn replace_history(mut self, messages: Vec<llm::Message>, now: OffsetDateTime) -> Self {
        self.messages = messages;
        self.draft_messages = Vec::new();
        self.compactions = Vec::new();
        self.history_generation = self.history_generation.saturating_add(1);
        self.version += 1;
        self.updated_at = now;
        self
    }
    /// Append a compaction entry without deleting or rewriting committed messages.
    ///
    /// `result_messages` is the compactor's effective output and
    /// `retained_message_count` is its unchanged trailing slice. Any recovery
    /// draft folded into the source view is first appended to the raw transcript,
    /// then cleared atomically with the compaction entry.
    ///
    /// # Errors
    /// Returns [`MessageProjectionError::CompactionSourceMismatch`] when the
    /// caller compacted a stale projection, or
    /// [`MessageProjectionError::InvalidCompactionRetention`] when the retained
    /// count exceeds the result length.
    pub fn append_compaction(
        mut self,
        result_messages: Vec<llm::Message>,
        source_message_count: usize,
        retained_message_count: usize,
        now: OffsetDateTime,
    ) -> Result<Self, MessageProjectionError> {
        let prior_boundary = self
            .compactions
            .last()
            .map_or(0, |entry| entry.compacted_end);
        let visible_raw_count = self.messages.len().saturating_sub(prior_boundary);
        let available = self
            .compactions
            .last()
            .map_or(0, |entry| entry.replacement_messages.len())
            .saturating_add(visible_raw_count)
            .saturating_add(self.draft_messages.len());
        if source_message_count != available {
            return Err(MessageProjectionError::CompactionSourceMismatch {
                available,
                provided: source_message_count,
            });
        }
        if retained_message_count > result_messages.len() {
            return Err(MessageProjectionError::InvalidCompactionRetention {
                retained: retained_message_count,
                result_count: result_messages.len(),
            });
        }

        let draft_fold_start = self.messages.len();
        let draft_fold_len = self.draft_messages.len();
        let draft_parked_len = parked_head_len(&self.draft_messages);
        self.messages.append(&mut self.draft_messages);
        let visible_raw_count = self.messages.len().saturating_sub(prior_boundary);
        let retained_raw_count = retained_message_count.min(visible_raw_count);
        let compacted_end = self.messages.len().saturating_sub(retained_raw_count);
        let replacement_count = result_messages.len().saturating_sub(retained_raw_count);
        let replacement_messages: Vec<_> = result_messages
            .into_iter()
            .take(replacement_count)
            .collect();
        let generated_summary = replacement_messages
            .first()
            .is_some_and(is_generated_compaction_summary);

        // Hard storage invariant (ENG-9651): the effective view this entry
        // selects must be a provider-valid tool sequence. An engine whose
        // boundary cuts an assistant `tool_use` / user `tool_result` pair
        // would otherwise persist a thread no provider accepts — invisible
        // to the pre-call repair, which validates the raw staged history.
        // Refuse before anything is recorded; the caller skips the
        // compaction and the turn proceeds uncompacted.
        let effective: Vec<llm::Message> = replacement_messages
            .iter()
            .cloned()
            .chain(self.messages[compacted_end..].iter().cloned())
            .collect();
        if let Some(index) = llm::provider_tool_sequence_error_index(&effective) {
            return Err(MessageProjectionError::CompactionBreaksToolSequence { index });
        }

        self.compactions.push(CompactionEntry {
            compacted_start: prior_boundary,
            compacted_end,
            replacement_messages,
            generated_summary,
            source_message_count,
            draft_fold_start,
            draft_fold_len,
            draft_parked_len,
            created_at: now,
        });
        self.history_generation = self.history_generation.saturating_add(1);
        self.version += 1;
        self.updated_at = now;
        Ok(self)
    }

    /// Append synthetic orphan-repair evidence while preserving raw history,
    /// then atomically select the balanced effective projection.
    ///
    /// # Errors
    /// Returns [`MessageProjectionError::EmptyCommit`] for empty evidence or
    /// [`MessageProjectionError::CompactionSourceMismatch`] for a stale repair.
    pub fn append_repair(
        mut self,
        mut repair_messages: Vec<llm::Message>,
        balanced_messages: Vec<llm::Message>,
        source_message_count: usize,
        now: OffsetDateTime,
    ) -> Result<Self, MessageProjectionError> {
        if repair_messages.is_empty() {
            return Err(MessageProjectionError::EmptyCommit);
        }
        let available = self.provider_view().len();
        if source_message_count != available {
            return Err(MessageProjectionError::CompactionSourceMismatch {
                available,
                provided: source_message_count,
            });
        }
        let prior_authoritative = self
            .compactions
            .last()
            .filter(|entry| entry.generated_summary)
            .and_then(|entry| entry.replacement_messages.first())
            .filter(|message| is_generated_compaction_summary(message))
            .cloned();
        let prior_boundary = self
            .compactions
            .last()
            .map_or(0, |entry| entry.compacted_end);
        let draft_fold_start = self.messages.len();
        let draft_fold_len = self.draft_messages.len();
        let draft_parked_len = parked_head_len(&self.draft_messages);
        self.messages.append(&mut self.draft_messages);
        self.messages.append(&mut repair_messages);
        let generated_summary = balanced_messages.first().is_some_and(|message| {
            prior_authoritative
                .as_ref()
                .is_some_and(|prior| prior == message)
                && is_generated_compaction_summary(message)
        });
        self.compactions.push(CompactionEntry {
            compacted_start: prior_boundary,
            compacted_end: self.messages.len(),
            replacement_messages: balanced_messages,
            generated_summary,
            source_message_count,
            draft_fold_start,
            draft_fold_len,
            draft_parked_len,
            created_at: now,
        });
        self.history_generation = self.history_generation.saturating_add(1);
        self.version += 1;
        self.updated_at = now;
        Ok(self)
    }

    /// The sequence the next provider call is built from: [`Self::context_history`]
    /// followed by the in-flight draft. [`Self::append_repair`] replaces exactly
    /// this view, so a repair must validate and count it, never the raw
    /// [`Self::messages`] transcript, which append-only repairs leave unchanged.
    #[must_use]
    pub fn provider_view(&self) -> Vec<llm::Message> {
        let mut view = self.context_history();
        view.extend(self.draft_messages.iter().cloned());
        view
    }

    /// A generated summary is authoritative only in the first replacement slot
    /// of the latest compaction entry and only while its complete shape remains
    /// valid. Typed markers in raw transcript messages, malformed Snapcompact
    /// messages, or later replacement slots are downgraded to ordinary text.
    #[must_use]
    pub fn context_history(&self) -> Vec<llm::Message> {
        let Some(entry) = self.compactions.last() else {
            return self
                .messages
                .iter()
                .cloned()
                .map(demote_untrusted_compaction_summary)
                .collect();
        };
        let mut history = Vec::with_capacity(
            entry
                .replacement_messages
                .len()
                .saturating_add(self.messages.len().saturating_sub(entry.compacted_end)),
        );
        history.extend(entry.replacement_messages.iter().cloned().enumerate().map(
            |(index, message)| {
                let is_authoritative = entry.generated_summary
                    && index == 0
                    && is_generated_compaction_summary(&message);
                if is_authoritative {
                    message
                } else {
                    demote_untrusted_compaction_summary(message)
                }
            },
        ));
        history.extend(
            self.messages[entry.compacted_end..]
                .iter()
                .cloned()
                .map(demote_untrusted_compaction_summary),
        );
        history
    }

    /// Where a compaction since the last commit folded `parked` — the
    /// `suspended_messages` a parked turn resumes from — into the raw
    /// transcript, or `None` when it has not.
    ///
    /// Every park writes its messages as the draft, and every resume persists
    /// them at the head of its draft before it may compact; a fold clears the
    /// draft. So parked messages the draft still leads with are not folded.
    /// Otherwise the fold holding them is the newest one since the last
    /// commit that recorded them as its parked head
    /// ([`CompactionEntry::draft_parked_len`]) and holds them verbatim.
    #[must_use]
    pub fn parked_fold(&self, parked: &[llm::Message]) -> Option<ParkedFold> {
        if parked.is_empty() || self.draft_messages.starts_with(parked) {
            return None;
        }
        let mut run_start = self.messages.len();
        for entry in self.compactions.iter().rev() {
            if entry.draft_fold_start.saturating_add(entry.draft_fold_len) != run_start {
                break;
            }
            run_start = entry.draft_fold_start;
            let holds_parked = entry.draft_parked_len == parked.len()
                && self
                    .messages
                    .get(run_start..run_start.saturating_add(parked.len()))
                    == Some(parked);
            if holds_parked {
                return Some(ParkedFold {
                    start: run_start,
                    len: parked.len(),
                });
            }
        }
        None
    }

    /// The raw messages folded after `fold`'s parked ones: what the attempts
    /// that resumed them staged before they died.
    #[must_use]
    pub fn after_parked(&self, fold: ParkedFold) -> &[llm::Message] {
        self.messages
            .get(fold.start.saturating_add(fold.len)..)
            .unwrap_or_default()
    }

    /// Replace the in-flight draft messages.
    ///
    /// Called at every tool-boundary suspension with the turn's messages
    /// that the raw transcript does not hold yet. The draft is overwritten
    /// (not appended) because each suspension carries that whole suffix.
    ///
    /// An empty `messages` argument is allowed and produces the same
    /// observable state as [`Self::clear_draft`], but
    /// [`Self::clear_draft`] is the canonical entry point for the
    /// turn-commit path.
    #[must_use]
    pub fn set_draft(mut self, messages: Vec<llm::Message>, now: OffsetDateTime) -> Self {
        self.draft_messages = messages;
        self.version += 1;
        self.updated_at = now;
        self
    }

    /// Drop the in-flight draft messages.
    ///
    /// Called after a successful turn commit so the next turn starts
    /// with no stale draft. No-op when the draft is already empty
    /// (still bumps the version so writers can observe the
    /// transition).
    #[must_use]
    pub fn clear_draft(mut self, now: OffsetDateTime) -> Self {
        self.draft_messages = Vec::new();
        self.version += 1;
        self.updated_at = now;
        self
    }

    /// Number of messages in the committed history.
    #[must_use]
    pub const fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// `true` when an in-flight draft is currently held.
    #[must_use]
    pub const fn has_draft(&self) -> bool {
        !self.draft_messages.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_id() -> ThreadId {
        ThreadId::from_string("t-msg-test")
    }

    fn canonical_snapcompact_message() -> llm::Message {
        let metadata = llm::SnapcompactMetadata {
            source_artifact_id: 42,
            truncated_chars: 1_024,
            frame_count: 2,
            frame_size: 1_932,
            source_len: None,
            source_sha256: None,
            frame_manifest: None,
        };
        llm::Message::user_with_content(vec![
            llm::ContentBlock::CompactionSummary {
                text: "source checkpoint".to_string(),
                artifact_ids: vec![0, 7, 42, 90, 91],
                snapcompact: Some(metadata),
            },
            llm::ContentBlock::CompactionSummary {
                text: "visible head".to_string(),
                artifact_ids: Vec::new(),
                snapcompact: None,
            },
            llm::ContentBlock::CompactionSummary {
                text: llm::SNAPCOMPACT_HISTORY_IMAGE_WARNING.to_string(),
                artifact_ids: Vec::new(),
                snapcompact: None,
            },
            llm::ContentBlock::Image {
                source: llm::ContentSource::new("image/png", "artifact://90")
                    .with_detail(llm::ImageDetail::High),
            },
            llm::ContentBlock::Image {
                source: llm::ContentSource::new("image/png", "artifact://91"),
            },
            llm::ContentBlock::CompactionSummary {
                text: "visible tail".to_string(),
                artifact_ids: Vec::new(),
                snapcompact: None,
            },
        ])
    }

    // ── construction ──────────────────────────────────────────────

    #[test]
    fn new_projection_is_empty_with_version_zero() {
        let p = MessageProjection::new(thread_id(), t0());
        assert_eq!(p.thread_id, thread_id());
        assert_eq!(p.messages, [] as [agent_sdk_foundation::Message; 0]);
        assert_eq!(p.version, 0);
        assert_eq!(p.message_count(), 0);
    }

    // ── append_committed ──────────────────────────────────────────

    #[test]
    fn append_committed_extends_history() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(
                vec![llm::Message::user("hello"), llm::Message::assistant("hi")],
                t_plus(1),
            )
            .expect("non-empty commit succeeds");
        assert_eq!(p.message_count(), 2);
        assert_eq!(p.version, 1);
        assert_eq!(p.updated_at, t_plus(1));
    }

    #[test]
    fn append_committed_accumulates_across_turns() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("turn 1")], t_plus(1))
            .unwrap();
        let p = p
            .append_committed(vec![llm::Message::assistant("reply 1")], t_plus(2))
            .unwrap();
        let p = p
            .append_committed(
                vec![
                    llm::Message::user("turn 2"),
                    llm::Message::assistant("reply 2"),
                ],
                t_plus(3),
            )
            .unwrap();
        assert_eq!(p.message_count(), 4);
        assert_eq!(p.version, 3);
    }

    #[test]
    fn append_committed_rejects_empty_batch() {
        let p = MessageProjection::new(thread_id(), t0());
        let err = p.append_committed(vec![], t_plus(1)).unwrap_err();
        assert_eq!(err, MessageProjectionError::EmptyCommit);
    }

    #[test]
    fn append_committed_preserves_order() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("first")], t_plus(1))
            .unwrap();
        let p = p
            .append_committed(vec![llm::Message::user("second")], t_plus(2))
            .unwrap();
        assert_eq!(p.messages[0].role, llm::Role::User);
        assert_eq!(p.messages[1].role, llm::Role::User);
        // Verify actual content ordering
        let texts: Vec<_> = p
            .messages
            .iter()
            .filter_map(|m| match &m.content {
                llm::Content::Text(t) => Some(t.as_str()),
                llm::Content::Blocks(_) => None,
            })
            .collect();
        assert_eq!(texts, vec!["first", "second"]);
    }

    // ── replace_history ───────────────────────────────────────────

    #[test]
    fn replace_history_swaps_entire_history() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(
                vec![
                    llm::Message::user("old 1"),
                    llm::Message::assistant("old 2"),
                    llm::Message::user("old 3"),
                ],
                t_plus(1),
            )
            .unwrap();
        assert_eq!(p.message_count(), 3);

        let p = p.replace_history(
            vec![
                llm::Message::user("[Summary]"),
                llm::Message::assistant("Continuing..."),
            ],
            t_plus(2),
        );
        assert_eq!(p.message_count(), 2);
        assert_eq!(p.version, 2); // append bumped to 1, replace bumps to 2
        assert_eq!(p.updated_at, t_plus(2));
    }

    #[test]
    fn replace_history_allows_empty_replacement() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("data")], t_plus(1))
            .unwrap();
        let p = p.replace_history(vec![], t_plus(2));
        assert_eq!(p.message_count(), 0);
        assert_eq!(p.version, 2);
    }

    #[test]
    fn replace_history_bumps_version() {
        let p = MessageProjection::new(thread_id(), t0());
        assert_eq!(p.version, 0);
        let p = p.replace_history(vec![llm::Message::user("fresh")], t_plus(1));
        assert_eq!(p.version, 1);
        let p = p.replace_history(vec![llm::Message::user("newer")], t_plus(2));
        assert_eq!(p.version, 2);
    }

    #[test]
    fn history_generation_counts_rewrite_boundaries_only() -> anyhow::Result<()> {
        let p = MessageProjection::new(thread_id(), t0());
        assert_eq!(p.history_generation, 0);

        let p = p.append_committed(vec![llm::Message::user("a")], t_plus(1))?;
        assert_eq!(
            p.history_generation, 0,
            "append-only commits are not rewrites"
        );

        let p = p.append_compaction(vec![llm::Message::user("summary")], 1, 0, t_plus(2))?;
        assert_eq!(p.history_generation, 1);
        assert_eq!(p.compactions.len(), 1);

        let p = p.replace_history(vec![llm::Message::user("rewritten")], t_plus(3));
        assert_eq!(p.compactions.len(), 0);
        assert_eq!(p.history_generation, 2);

        let repaired = p.append_repair(
            vec![llm::Message::user("evidence")],
            vec![llm::Message::user("balanced")],
            1,
            t_plus(4),
        )?;
        assert_eq!(repaired.history_generation, 3);
        Ok(())
    }

    // ── version monotonicity ──────────────────────────────────────

    #[test]
    fn version_increases_monotonically_across_mixed_operations() {
        let p = MessageProjection::new(thread_id(), t0());
        assert_eq!(p.version, 0);

        let p = p
            .append_committed(vec![llm::Message::user("a")], t_plus(1))
            .unwrap();
        assert_eq!(p.version, 1);

        let p = p.replace_history(vec![llm::Message::user("b")], t_plus(2));
        assert_eq!(p.version, 2);

        let p = p
            .append_committed(vec![llm::Message::user("c")], t_plus(3))
            .unwrap();
        assert_eq!(p.version, 3);
    }

    // ── identity ──────────────────────────────────────────────────

    #[test]
    fn thread_id_and_created_at_survive_mutations() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("a")], t_plus(1))
            .unwrap();
        let p = p.replace_history(vec![], t_plus(2));
        assert_eq!(p.thread_id, thread_id());
        assert_eq!(p.created_at, t0());
    }

    // ── set_draft / clear_draft ───────────────────────────────────

    #[test]
    fn new_projection_has_no_draft() {
        let p = MessageProjection::new(thread_id(), t0());
        assert!(!p.has_draft());
        assert_eq!(p.draft_messages, [] as [agent_sdk_foundation::Message; 0]);
    }

    #[test]
    fn set_draft_stores_in_flight_messages_without_touching_committed() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("committed")], t_plus(1))
            .unwrap();
        assert_eq!(p.message_count(), 1);
        assert_eq!(p.version, 1);

        let p = p.set_draft(
            vec![
                llm::Message::user("turn 2 prompt"),
                llm::Message::assistant("calling tool"),
            ],
            t_plus(2),
        );
        // Committed history is untouched.
        assert_eq!(p.message_count(), 1);
        // Draft now carries the in-flight messages.
        assert!(p.has_draft());
        assert_eq!(p.draft_messages.len(), 2);
        assert_eq!(p.version, 2);
        assert_eq!(p.updated_at, t_plus(2));
    }

    #[test]
    fn set_draft_overwrites_prior_draft() {
        // Each suspension passes the full accumulated suspended_messages,
        // so set_draft must replace the slot rather than append.
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.set_draft(vec![llm::Message::user("first suspension")], t_plus(1));
        let p = p.set_draft(
            vec![
                llm::Message::user("first suspension"),
                llm::Message::assistant("second tool boundary"),
            ],
            t_plus(2),
        );
        assert_eq!(p.draft_messages.len(), 2);
        assert_eq!(p.version, 2);
    }

    #[test]
    fn clear_draft_drops_in_flight_messages() {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.set_draft(vec![llm::Message::user("in flight")], t_plus(1));
        assert!(p.has_draft());
        let p = p.clear_draft(t_plus(2));
        assert!(!p.has_draft());
        assert_eq!(p.version, 2);
        assert_eq!(p.updated_at, t_plus(2));
    }

    #[test]
    fn clear_draft_on_empty_slot_still_bumps_version() {
        // The store path may call clear_draft unconditionally on every
        // commit — calling it with an already-empty slot must remain
        // observable so optimistic-concurrency writers don't miss the
        // transition.
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.clear_draft(t_plus(1));
        assert!(!p.has_draft());
        assert_eq!(p.version, 1);
    }

    #[test]
    fn append_committed_preserves_draft_but_replace_history_clears_it() {
        // `append_committed` operates on the committed slot only, so the
        // draft persists verbatim across it. `replace_history` is a full
        // swap of the committed history that callers reach only after
        // folding the draft into the new history — so it clears the draft
        // in the same operation, closing the crash window where a separate
        // clear could fail and leave a draft that double-folds on recover.
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.set_draft(vec![llm::Message::user("draft")], t_plus(1));

        let p = p
            .append_committed(vec![llm::Message::user("committed-1")], t_plus(2))
            .unwrap();
        assert!(p.has_draft(), "append_committed must preserve the draft");
        assert_eq!(p.draft_messages.len(), 1);

        let p = p.replace_history(vec![llm::Message::user("[summary]")], t_plus(3));
        assert!(
            !p.has_draft(),
            "replace_history must clear the draft atomically",
        );
    }

    #[test]
    fn draft_round_trips_through_json() -> anyhow::Result<()> {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.set_draft(
            vec![
                llm::Message::user("draft prompt"),
                llm::Message::assistant("draft asst"),
            ],
            t_plus(1),
        );
        let json = serde_json::to_string(&p)?;
        let recovered: MessageProjection = serde_json::from_str(&json)?;
        assert!(recovered.has_draft());
        assert_eq!(recovered.draft_messages.len(), 2);
        assert_eq!(recovered.version, p.version);
        Ok(())
    }

    #[test]
    fn legacy_json_without_draft_field_decodes_empty() -> anyhow::Result<()> {
        // Older daemons serialized rows before draft_messages existed.
        // The field is `#[serde(default)]` so legacy payloads must
        // still round-trip with an empty draft slot.
        let legacy = serde_json::json!({
            "thread_id": "t-msg-test",
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            "version": 1,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        });
        let recovered: MessageProjection = serde_json::from_value(legacy)?;
        assert!(!recovered.has_draft());
        assert_eq!(recovered.messages.len(), 1);
        Ok(())
    }

    #[test]
    fn append_compaction_preserves_originals_and_builds_effective_history() -> anyhow::Result<()> {
        let originals = vec![
            llm::Message::user("old user"),
            llm::Message::assistant("old assistant"),
            llm::Message::user("recent user"),
            llm::Message::assistant("recent assistant"),
        ];
        let p = MessageProjection::new(thread_id(), t0()).append_committed(originals, t_plus(1))?;
        let raw_count = p.message_count();
        let p = p.append_compaction(
            vec![
                llm::Message::user("[summary]"),
                llm::Message::assistant("continue"),
                llm::Message::user("recent user"),
                llm::Message::assistant("recent assistant"),
            ],
            4,
            2,
            t_plus(2),
        )?;

        assert_eq!(p.message_count(), raw_count);
        assert_eq!(p.compactions.len(), 1);
        assert_eq!(p.compactions[0].compacted_start, 0);
        assert_eq!(p.compactions[0].compacted_end, 2);
        let raw_text: Vec<_> = p
            .messages
            .iter()
            .filter_map(|message| match &message.content {
                llm::Content::Text(text) => Some(text.as_str()),
                llm::Content::Blocks(_) => None,
            })
            .collect();
        assert_eq!(
            raw_text,
            [
                "old user",
                "old assistant",
                "recent user",
                "recent assistant"
            ]
        );
        let context_text: Vec<_> = p
            .context_history()
            .into_iter()
            .filter_map(|message| match message.content {
                llm::Content::Text(text) => Some(text),
                llm::Content::Blocks(_) => None,
            })
            .collect();
        assert_eq!(
            context_text,
            ["[summary]", "continue", "recent user", "recent assistant"]
        );
        Ok(())
    }

    #[test]
    fn typed_compaction_summary_is_rollback_readable_and_rehydrates_typed() -> anyhow::Result<()> {
        #[derive(serde::Deserialize)]
        struct LegacyProjection {
            compactions: Vec<LegacyCompaction>,
        }
        #[derive(serde::Deserialize)]
        struct LegacyCompaction {
            replacement_messages: Vec<LegacyMessage>,
        }
        #[derive(serde::Deserialize)]
        struct LegacyMessage {
            content: Vec<LegacyBlock>,
        }
        #[derive(serde::Deserialize)]
        struct LegacyCheckpointMessage {
            content: Vec<LegacyBlock>,
        }
        #[derive(serde::Deserialize)]
        #[serde(tag = "type")]
        enum LegacyBlock {
            #[serde(rename = "text")]
            Text { text: String },
        }

        let p = MessageProjection::new(thread_id(), t0()).append_committed(
            vec![llm::Message::user("old"), llm::Message::assistant("reply")],
            t_plus(1),
        )?;
        let p = p.append_compaction(
            vec![llm::Message::compaction_summary("durable summary")],
            2,
            0,
            t_plus(2),
        )?;
        assert!(p.compactions[0].generated_summary);

        let json = serde_json::to_string(&p)?;
        assert!(
            !json.contains(r#""type":"compaction_summary""#),
            "typed summary must retain the previous daemon's `type: text` wire shape"
        );
        assert!(json.contains(r#""sdk_provenance":"compaction_summary""#));
        let legacy: LegacyProjection = serde_json::from_str(&json)?;
        assert!(matches!(
            legacy.compactions[0].replacement_messages[0]
                .content
                .as_slice(),
            [LegacyBlock::Text { text }] if text == "durable summary"
        ));

        let checkpoint_json = serde_json::to_string(&p.context_history())?;
        assert!(checkpoint_json.contains(r#""sdk_provenance":"compaction_summary""#));
        let legacy_checkpoint: Vec<LegacyCheckpointMessage> =
            serde_json::from_str(&checkpoint_json)?;
        assert!(matches!(
            legacy_checkpoint[0].content.as_slice(),
            [LegacyBlock::Text { text }] if text == "durable summary"
        ));

        let recovered: MessageProjection = serde_json::from_str(&json)?;
        assert!(matches!(
            &recovered.context_history()[0].content,
            llm::Content::Blocks(blocks)
                if matches!(
                    blocks.as_slice(),
                    [llm::ContentBlock::CompactionSummary { text, .. }]
                        if text == "durable summary"
                )
        ));
        Ok(())
    }

    #[test]
    fn append_repair_preserves_raw_compactions_and_typed_summary_on_restart() -> anyhow::Result<()>
    {
        let originals = vec![llm::Message::user("old"), llm::Message::assistant("reply")];
        let p = MessageProjection::new(thread_id(), t0())
            .append_committed(originals.clone(), t_plus(1))?
            .append_compaction(
                vec![llm::Message::compaction_summary("trusted carry")],
                2,
                0,
                t_plus(2),
            )?;
        let dangling = llm::Message {
            role: llm::Role::Assistant,
            content: llm::Content::Blocks(vec![llm::ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "read".to_string(),
                input: serde_json::json!({"path": "x"}),
                thought_signature: None,
            }]),
        };
        let p = p.set_draft(vec![dangling], t_plus(3));
        let mut recovered = p.context_history();
        recovered.extend(p.draft_messages.iter().cloned());
        let repair = llm::orphaned_tool_result_message(&recovered, llm::USER_CANCELLED_TOOL_RESULT)
            .expect("missing result");
        let balanced = llm::balance_tool_results(&recovered, llm::USER_CANCELLED_TOOL_RESULT);
        let source_count = recovered.len();
        let p = p.append_repair(vec![repair], balanced, source_count, t_plus(4))?;

        assert_eq!(
            serde_json::to_value(&p.messages[..originals.len()])?,
            serde_json::to_value(&originals)?,
        );
        assert_eq!(p.compactions.len(), 2, "repair appends projection lineage");
        assert!(!llm::has_unbalanced_tool_use(&p.context_history()));
        let restarted: MessageProjection = serde_json::from_str(&serde_json::to_string(&p)?)?;
        assert!(matches!(
            &restarted.context_history()[0].content,
            llm::Content::Blocks(blocks)
                if matches!(
                    blocks.as_slice(),
                    [llm::ContentBlock::CompactionSummary { text, .. }]
                        if text == "trusted carry"
                )
        ));
        Ok(())
    }

    #[test]
    fn snapcompact_survives_append_reload_and_exact_repair_forwarding() -> anyhow::Result<()> {
        let snapcompact = canonical_snapcompact_message();
        let p = MessageProjection::new(thread_id(), t0())
            .append_committed(
                vec![llm::Message::user("old"), llm::Message::assistant("reply")],
                t_plus(1),
            )?
            .append_compaction(vec![snapcompact.clone()], 2, 0, t_plus(2))?
            .append_committed(vec![llm::Message::user("after compaction")], t_plus(3))?;
        assert!(
            p.compactions
                .last()
                .context("missing Snapcompact entry")?
                .generated_summary
        );

        let serialized = serde_json::to_string(&p)?;
        let reloaded: MessageProjection = serde_json::from_str(&serialized)?;
        let history = reloaded.context_history();
        let first = history.first().context("missing Snapcompact replacement")?;
        assert_eq!(first, &snapcompact);
        let llm::Content::Blocks(blocks) = &first.content else {
            anyhow::bail!("Snapcompact replacement lost its block content");
        };
        assert!(matches!(
            blocks.first(),
            Some(llm::ContentBlock::CompactionSummary {
                artifact_ids,
                snapcompact: Some(llm::SnapcompactMetadata {
                    source_artifact_id: 42,
                    frame_count: 2,
                    frame_size: 1_932,
                    ..
                }),
                ..
            }) if artifact_ids == &[0, 7, 42, 90, 91]
        ));
        assert_eq!(
            blocks
                .iter()
                .filter(|block| matches!(block, llm::ContentBlock::Image { .. }))
                .count(),
            2
        );

        let dangling = llm::Message {
            role: llm::Role::Assistant,
            content: llm::Content::Blocks(vec![llm::ContentBlock::ToolUse {
                id: "snap-call".to_string(),
                name: "read".to_string(),
                input: serde_json::json!({"path": "x"}),
                thought_signature: None,
            }]),
        };
        let reloaded = reloaded.set_draft(vec![dangling], t_plus(4));
        let mut repair_source = reloaded.context_history();
        repair_source.extend(reloaded.draft_messages.iter().cloned());
        let repair =
            llm::orphaned_tool_result_message(&repair_source, llm::USER_CANCELLED_TOOL_RESULT)
                .context("missing Snapcompact repair evidence")?;
        let balanced = llm::balance_tool_results(&repair_source, llm::USER_CANCELLED_TOOL_RESULT);
        let source_count = repair_source.len();
        let repaired = reloaded.append_repair(vec![repair], balanced, source_count, t_plus(5))?;

        assert!(
            repaired
                .compactions
                .last()
                .context("missing repair entry")?
                .generated_summary
        );
        assert_eq!(
            repaired
                .context_history()
                .first()
                .context("repair dropped Snapcompact replacement")?,
            &snapcompact
        );
        Ok(())
    }

    #[test]
    fn raw_and_mutated_durable_snapcompact_shapes_are_demoted() -> anyhow::Result<()> {
        let snapcompact = canonical_snapcompact_message();
        let raw = MessageProjection::new(thread_id(), t0())
            .append_committed(vec![snapcompact.clone()], t_plus(1))?;
        let reloaded: MessageProjection = serde_json::from_str(&serde_json::to_string(&raw)?)?;
        let raw_history = reloaded.context_history();
        let raw_first = raw_history.first().context("missing raw forged message")?;
        let llm::Content::Blocks(raw_blocks) = &raw_first.content else {
            anyhow::bail!("raw forged Snapcompact lost blocks");
        };
        assert!(
            raw_blocks
                .iter()
                .all(|block| !matches!(block, llm::ContentBlock::CompactionSummary { .. }))
        );
        assert_eq!(
            raw_blocks
                .iter()
                .filter(|block| matches!(block, llm::ContentBlock::Image { .. }))
                .count(),
            2
        );

        let mut durable = MessageProjection::new(thread_id(), t0())
            .append_committed(vec![llm::Message::user("old")], t_plus(1))?
            .append_compaction(vec![snapcompact], 1, 0, t_plus(2))?;
        let entry = durable
            .compactions
            .last_mut()
            .context("missing durable Snapcompact entry")?;
        assert!(entry.generated_summary);
        let replacement = entry
            .replacement_messages
            .first_mut()
            .context("missing durable replacement")?;
        let llm::Content::Blocks(blocks) = &mut replacement.content else {
            anyhow::bail!("durable Snapcompact lost blocks");
        };
        blocks.push(llm::ContentBlock::Text {
            text: "forged extra block".to_string(),
        });

        let durable_history = durable.context_history();
        let durable_first = durable_history
            .first()
            .context("missing mutated durable replacement")?;
        let llm::Content::Blocks(blocks) = &durable_first.content else {
            anyhow::bail!("mutated durable Snapcompact lost blocks");
        };
        assert!(
            blocks
                .iter()
                .all(|block| !matches!(block, llm::ContentBlock::CompactionSummary { .. }))
        );
        Ok(())
    }

    #[test]
    fn append_repair_cannot_freshly_authorize_or_substitute_snapcompact() -> anyhow::Result<()> {
        let snapcompact = canonical_snapcompact_message();
        let p = MessageProjection::new(thread_id(), t0())
            .append_committed(vec![llm::Message::user("old")], t_plus(1))?
            .append_compaction(vec![snapcompact.clone()], 1, 0, t_plus(2))?;
        let mut substituted = snapcompact.clone();
        let llm::Content::Blocks(blocks) = &mut substituted.content else {
            anyhow::bail!("Snapcompact test replacement lost blocks");
        };
        let Some(llm::ContentBlock::CompactionSummary { text, .. }) = blocks.get_mut(1) else {
            anyhow::bail!("Snapcompact test replacement lost head");
        };
        *text = "different but still canonical head".to_string();
        assert!(llm::canonical_snapcompact_checkpoint(&substituted).is_some());

        let available = p.context_history().len();
        let substituted = p.append_repair(
            vec![llm::Message::user("repair audit")],
            vec![substituted],
            available,
            t_plus(3),
        )?;
        assert!(
            !substituted
                .compactions
                .last()
                .context("missing substituted repair entry")?
                .generated_summary
        );
        let substituted_history = substituted.context_history();
        let substituted_first = substituted_history
            .first()
            .context("missing substituted repair replacement")?;
        assert!(matches!(
            &substituted_first.content,
            llm::Content::Blocks(blocks)
                if blocks
                    .iter()
                    .all(|block| !matches!(
                        block,
                        llm::ContentBlock::CompactionSummary { .. }
                    ))
        ));

        let fresh = MessageProjection::new(thread_id(), t0())
            .append_committed(vec![llm::Message::user("raw")], t_plus(1))?
            .append_repair(
                vec![llm::Message::user("repair audit")],
                vec![snapcompact],
                1,
                t_plus(2),
            )?;
        assert!(
            !fresh
                .compactions
                .last()
                .context("missing fresh repair entry")?
                .generated_summary
        );
        assert!(matches!(
            fresh.context_history().first(),
            Some(llm::Message {
                content: llm::Content::Blocks(blocks),
                ..
            }) if blocks
                .iter()
                .all(|block| !matches!(
                    block,
                    llm::ContentBlock::CompactionSummary { .. }
                ))
        ));
        Ok(())
    }

    /// ENG-9651: a compaction whose retained tail starts with a dangling
    /// `tool_result` (the boundary cut the use/result pair) is refused
    /// before anything is recorded, and the projection is left untouched.
    #[test]
    fn compaction_that_cuts_a_tool_pair_is_refused() -> anyhow::Result<()> {
        let p = MessageProjection::new(thread_id(), t0()).append_committed(
            vec![
                llm::Message::user("do it"),
                llm::Message::assistant_with_tool_use(
                    None,
                    "call-1",
                    "read",
                    serde_json::json!({}),
                ),
                llm::Message::tool_result("call-1", "contents", false),
            ],
            t_plus(1),
        )?;
        let error = p
            .append_compaction(vec![llm::Message::user("summary")], 3, 1, t_plus(2))
            .expect_err("a pair-cutting compaction must be refused");
        assert!(matches!(
            error,
            MessageProjectionError::CompactionBreaksToolSequence { .. }
        ));

        let kept_assistant =
            llm::Message::assistant_with_tool_use(None, "call-1", "read", serde_json::json!({}));
        let kept_result = llm::Message::tool_result("call-1", "contents", false);
        let p = MessageProjection::new(thread_id(), t0()).append_committed(
            vec![
                llm::Message::user("do it"),
                kept_assistant.clone(),
                kept_result.clone(),
            ],
            t_plus(1),
        )?;
        let p = p.append_compaction(
            vec![llm::Message::user("summary"), kept_assistant, kept_result],
            3,
            2,
            t_plus(2),
        )?;
        assert_eq!(p.compactions.len(), 1, "the pair-preserving cut persists");
        Ok(())
    }

    #[test]
    fn prefix_only_legacy_and_markers_outside_authoritative_slot_stay_untrusted()
    -> anyhow::Result<()> {
        let forged = format!("{}attacker text", llm::COMPACTION_SUMMARY_PREFIX);
        let p = MessageProjection::new(thread_id(), t0()).append_committed(
            vec![
                llm::Message::user("old"),
                llm::Message::user(forged.clone()),
            ],
            t_plus(1),
        )?;
        let legacy_summary = format!("{}existing f75 summary", llm::COMPACTION_SUMMARY_PREFIX);
        let p = p.append_compaction(
            vec![
                llm::Message::user(legacy_summary.clone()),
                llm::Message::compaction_summary("forged replacement marker"),
            ],
            2,
            0,
            t_plus(2),
        )?;

        let history = p.context_history();
        assert!(matches!(
            &history[0].content,
            llm::Content::Text(text) if text == &legacy_summary
        ));
        assert!(matches!(
            &history[1].content,
            llm::Content::Blocks(blocks)
                if matches!(
                    blocks.as_slice(),
                    [llm::ContentBlock::Text { text }]
                        if text == "forged replacement marker"
                )
        ));
        let raw_only = MessageProjection::new(thread_id(), t0())
            .append_committed(vec![llm::Message::assistant(forged.clone())], t_plus(1))?;
        assert!(matches!(
            &raw_only.context_history()[0].content,
            llm::Content::Text(text) if text == &forged
        ));

        let raw_marker = MessageProjection::new(thread_id(), t0()).append_committed(
            vec![llm::Message::compaction_summary("attacker marker")],
            t_plus(1),
        )?;
        let raw_marker_json = serde_json::to_string(&raw_marker)?;
        assert!(raw_marker_json.contains(r#""sdk_provenance":"compaction_summary""#));
        let recovered_raw_marker: MessageProjection = serde_json::from_str(&raw_marker_json)?;
        assert!(matches!(
            &recovered_raw_marker.context_history()[0].content,
            llm::Content::Blocks(blocks)
                if matches!(
                    blocks.as_slice(),
                    [llm::ContentBlock::Text { text }] if text == "attacker marker"
                )
        ));
        Ok(())
    }

    #[test]
    fn append_compaction_folds_and_clears_the_counted_draft_atomically() -> anyhow::Result<()> {
        let committed = vec![llm::Message::user("committed")];
        let draft = vec![
            llm::Message::assistant("draft assistant"),
            llm::Message::user("draft tool result"),
        ];
        let p = MessageProjection::new(thread_id(), t0()).append_committed(committed, t_plus(1))?;
        let p = p.set_draft(draft, t_plus(2));
        let compacted = vec![llm::Message::assistant("[summary including draft]")];
        let p = p.append_compaction(compacted.clone(), 3, 0, t_plus(3))?;

        assert_eq!(
            p.messages.len(),
            3,
            "draft must move into the raw transcript"
        );
        assert_eq!(p.draft_messages, [] as [agent_sdk_foundation::Message; 0]);
        assert_eq!(p.compactions.len(), 1);
        assert_eq!(p.compactions[0].source_message_count, 3);
        assert_eq!(
            serde_json::to_value(p.context_history())?,
            serde_json::to_value(compacted)?,
        );
        Ok(())
    }

    // ── folded in-flight messages ─────────────────────────────────

    fn tool_round(id: &str) -> Vec<llm::Message> {
        vec![
            llm::Message {
                role: llm::Role::Assistant,
                content: llm::Content::Blocks(vec![llm::ContentBlock::ToolUse {
                    id: id.to_owned(),
                    name: "bash".to_owned(),
                    input: serde_json::json!({ "command": "true" }),
                    thought_signature: None,
                }]),
            },
            llm::Message::user_with_content(vec![llm::ContentBlock::ToolResult {
                tool_use_id: id.to_owned(),
                content: "ok".to_owned(),
                artifact: None,
                is_error: None,
            }]),
        ]
    }

    fn summary() -> Vec<llm::Message> {
        vec![llm::Message::user("[summary]")]
    }

    /// `[prompt, call_1 tool_use, call_1 tool_result]`, folded by one
    /// compaction after two committed messages.
    fn projection_with_folded_turn() -> anyhow::Result<(MessageProjection, Vec<llm::Message>)> {
        let mut in_flight = vec![llm::Message::user("prompt")];
        in_flight.extend(tool_round("call_1"));
        let p = MessageProjection::new(thread_id(), t0())
            .append_committed(
                vec![
                    llm::Message::user("earlier"),
                    llm::Message::assistant("reply"),
                ],
                t_plus(1),
            )?
            .set_draft(in_flight.clone(), t_plus(2))
            .append_compaction(summary(), 5, 0, t_plus(3))?;
        Ok((p, in_flight))
    }

    #[test]
    fn append_compaction_records_the_raw_range_and_parked_head_it_folded() -> anyhow::Result<()> {
        let (p, in_flight) = projection_with_folded_turn()?;
        let entry = p.compactions.last().context("compaction entry")?;
        assert_eq!(
            (
                entry.draft_fold_start,
                entry.draft_fold_len,
                entry.draft_parked_len
            ),
            (2, 3, 2),
            "the parked head runs through the draft's last assistant message",
        );
        assert_eq!(p.messages[2..], in_flight[..]);

        let repaired = p.set_draft(tool_round("call_2"), t_plus(4)).append_repair(
            vec![llm::Message::user("repair evidence")],
            summary(),
            3,
            t_plus(5),
        )?;
        let entry = repaired.compactions.last().context("repair entry")?;
        assert_eq!(
            (
                entry.draft_fold_start,
                entry.draft_fold_len,
                entry.draft_parked_len
            ),
            (5, 2, 1),
        );

        let parked_nothing = repaired
            .set_draft(vec![llm::Message::user("late result")], t_plus(6))
            .append_compaction(summary(), 2, 0, t_plus(7))?;
        let entry = parked_nothing.compactions.last().context("fold entry")?;
        assert_eq!(
            entry.draft_parked_len, 0,
            "a draft without an assistant message parked nothing",
        );
        Ok(())
    }

    #[test]
    fn parked_messages_are_held_only_by_the_fold_that_recorded_them() -> anyhow::Result<()> {
        let (p, in_flight) = projection_with_folded_turn()?;
        let parked = &in_flight[..2];
        let fold = p
            .parked_fold(parked)
            .context("the fold holds the parked messages")?;
        assert_eq!(fold, ParkedFold { start: 2, len: 2 });
        assert_eq!(
            p.after_parked(fold),
            &in_flight[2..],
            "the dead attempt's fan-in follows them",
        );

        assert_eq!(
            p.parked_fold(&in_flight[..1]),
            None,
            "a fold that parked two messages does not hold a one-message park",
        );
        assert_eq!(p.parked_fold(&in_flight), None, "nor a three-message one");
        assert_eq!(p.parked_fold(&tool_round("call_9")), None);
        assert_eq!(p.parked_fold(&[]), None);

        let committed = p.append_committed(vec![llm::Message::assistant("final")], t_plus(4))?;
        assert_eq!(
            committed.parked_fold(parked),
            None,
            "a commit ends the in-flight run",
        );
        Ok(())
    }

    #[test]
    fn a_rerun_fold_extends_what_follows_the_parked_messages() -> anyhow::Result<()> {
        let (p, in_flight) = projection_with_folded_turn()?;
        let late = llm::Message::user("late result");
        let p = p
            .set_draft(vec![late.clone()], t_plus(4))
            .append_compaction(summary(), 2, 0, t_plus(5))?;

        let fold = p
            .parked_fold(&in_flight[..2])
            .context("the first fold still holds the parked messages")?;
        assert_eq!(fold, ParkedFold { start: 2, len: 2 });
        let mut after = in_flight[2..].to_vec();
        after.push(late);
        assert_eq!(p.after_parked(fold), &after[..]);
        Ok(())
    }

    /// Two steering wakes answer verbatim alike. A fold holds the first
    /// answer as its parked head; the second answer, parked after that fold,
    /// is the draft and must not pass for the folded one.
    #[test]
    fn a_reply_parked_after_an_identical_folded_one_is_not_folded() -> anyhow::Result<()> {
        let (p, _) = projection_with_folded_turn()?;
        let answer = llm::Message::assistant("Still working on it.");
        let p = p
            .set_draft(
                vec![answer.clone(), llm::Message::user("how is it going?")],
                t_plus(4),
            )
            .append_compaction(summary(), 3, 0, t_plus(5))?;
        let reparked = std::slice::from_ref(&answer);
        assert!(
            p.parked_fold(reparked).is_some(),
            "precondition: a fold holds an identical parked head",
        );

        let p = p.set_draft(reparked.to_vec(), t_plus(6));
        assert_eq!(p.parked_fold(reparked), None);
        Ok(())
    }

    /// A turn that ended without committing leaves its fold in the run; the
    /// next turn, parked with the same prompt and reply, is not that fold.
    #[test]
    fn a_later_turn_parked_like_an_uncommitted_fold_is_not_folded() -> anyhow::Result<()> {
        let parked = vec![
            llm::Message::user("status?"),
            llm::Message::assistant("Still working on it."),
        ];
        let mut draft = parked.clone();
        draft.push(llm::Message::user("late result"));
        let p = MessageProjection::new(thread_id(), t0())
            .set_draft(draft, t_plus(1))
            .append_compaction(summary(), 3, 0, t_plus(2))?;
        assert!(
            p.parked_fold(&parked).is_some(),
            "precondition: the uncommitted fold holds that parked head",
        );

        let p = p.set_draft(parked.clone(), t_plus(3));
        assert_eq!(p.parked_fold(&parked), None);
        Ok(())
    }

    #[test]
    fn folds_without_a_recorded_parked_head_or_behind_a_repair_hold_nothing() -> anyhow::Result<()>
    {
        let (p, in_flight) = projection_with_folded_turn()?;
        let parked = &in_flight[..2];

        for missing in [
            &["draft_parked_len"][..],
            &["draft_fold_start", "draft_fold_len", "draft_parked_len"],
        ] {
            let mut legacy = serde_json::to_value(&p)?;
            for entry in legacy["compactions"]
                .as_array_mut()
                .context("compactions array")?
            {
                let fields = entry.as_object_mut().context("compaction entry object")?;
                for field in missing {
                    fields.remove(*field);
                }
            }
            let legacy: MessageProjection = serde_json::from_value(legacy)?;
            assert_eq!(
                legacy.parked_fold(parked),
                None,
                "rows written without {missing:?} carry no evidence",
            );
        }

        let repaired = p.append_repair(
            vec![llm::Message::user("repair evidence")],
            summary(),
            1,
            t_plus(4),
        )?;
        assert_eq!(repaired.parked_fold(parked), None);
        Ok(())
    }

    // ── wire format ───────────────────────────────────────────────

    #[test]
    fn projection_round_trips_through_json() -> anyhow::Result<()> {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p.append_committed(
            vec![llm::Message::user("hello"), llm::Message::assistant("hi")],
            t_plus(1),
        )?;
        let json = serde_json::to_string(&p)?;
        let recovered: MessageProjection = serde_json::from_str(&json)?;
        assert_eq!(recovered.thread_id, p.thread_id);
        assert_eq!(recovered.version, p.version);
        assert_eq!(recovered.messages.len(), p.messages.len());
        Ok(())
    }

    #[test]
    fn wire_format_keys_are_stable() -> anyhow::Result<()> {
        let p = MessageProjection::new(thread_id(), t0());
        let p = p
            .append_committed(vec![llm::Message::user("test")], t_plus(1))
            .unwrap();
        let value = serde_json::to_value(&p)?;
        for key in [
            "thread_id",
            "messages",
            "version",
            "created_at",
            "updated_at",
        ] {
            assert!(
                value.get(key).is_some(),
                "MessageProjection wire format lost key `{key}` — this breaks the server contract"
            );
        }
        Ok(())
    }
}
