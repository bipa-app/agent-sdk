//! Multi-modal user input for root-turn entry.
//!
//! Until this module landed, [`super::root_turn::execute_root_turn`]
//! took `user_prompt: &str` — a single string the worker stuffed into
//! `Message::user(text)` before calling the LLM. That works for
//! text-only conversations but discards the binary attachments the
//! gRPC submit RPC already accepts:
//!
//! ```text
//! message UserInputItem {
//!   oneof item {
//!     string text = 1;
//!     BinaryAttachment image = 2;
//!     BinaryAttachment document = 3;
//!   }
//! }
//! ```
//!
//! The host (`agent-service-host`) admitted those attachments into
//! `task.submitted_input` correctly, but the worker's
//! `root_task_prompt` helper rejected anything other than text with
//! `"root task input item is not supported by the service host yet"`,
//! so any thread carrying an `Image` or `Document` failed
//! `submit_thread_work` outright. This module replaces that string-only
//! pipeline with a typed [`UserInput`] that flows the original blocks
//! all the way to `Message::user_with_content(blocks)` — and survives
//! the auto-compaction recovery path that re-builds the chat request
//! after a `prompt is too long` error.
//!
//! # Audit-string projection
//!
//! `LlmRetryParams` and `TurnAttempt::request_blob` both store a
//! free-form `user_prompt` string for audit and resume detection. The
//! `Image`/`Document` blocks have no good text projection, so
//! [`UserInput::audit_summary`] concatenates the text-block content
//! and replaces every binary block with a stable
//! `[<media_type> attachment]` placeholder. Replay tooling that
//! examines the audit row sees something descriptive without needing
//! to round-trip the bytes.

use agent_sdk_foundation::llm::{ContentBlock, ContentSource, Message};

use crate::journal::task::SubmittedInputItem;

/// Resume placeholder for the worker's audit blob.
///
/// Synthetic `attempt_user_prompt` value used by `resume_root_turn`
/// when opening retry attempts so audit records can distinguish a
/// fresh-turn retry from a resume retry. Shared as a constant so the
/// worker and the resume path never drift.
pub const RESUME_AUDIT_PROMPT: &str = "<resume>";

/// Boundary-injection placeholder for the worker's audit blob.
///
/// Synthetic `attempt_user_prompt` value used when a turn opens an extra LLM
/// call to deliver input that arrived mid-turn. Distinct from
/// [`RESUME_AUDIT_PROMPT`] so audit readers can tell a resume-from-children
/// call apart from a steering injection, and a constant for the same reason
/// that one is: the sentinel consumers grep for must be the sentinel written.
pub const BOUNDARY_INJECTION_AUDIT_PROMPT: &str = "<boundary-injection>";

/// Typed user input for a root-turn entry.
///
/// Wraps the [`ContentBlock`] vector that the worker hands to
/// `Message::user_with_content(blocks)` together with helpers that
/// derive the audit-friendly string projections the journal expects.
/// `Vec<ContentBlock>` directly would have been the shortest path,
/// but a dedicated type lets us:
///
/// * Convert `&str` test fixtures via `From` without touching the
///   call sites' ergonomics (`execute_root_turn(inputs, "hi".into(), …)`).
/// * Keep the `&[ContentBlock]` view (`blocks()`) and the audit
///   projection (`audit_summary()`) consistent — derive the audit
///   string once, share the slice everywhere else.
/// * Carry a synthetic `Resume` variant so the resume path doesn't
///   need a sentinel string in the type system.
#[derive(Clone, Debug)]
pub struct UserInput {
    blocks: Vec<ContentBlock>,
    /// Pre-computed audit string. `Some` when the input was constructed
    /// via [`UserInput::resume`] (so we surface the
    /// `RESUME_AUDIT_PROMPT` sentinel directly), `None` otherwise (so
    /// we lazily flatten text blocks on demand).
    audit_override: Option<&'static str>,
}

impl UserInput {
    /// Construct from a single text string. Convenience for the
    /// common case (and every existing test fixture that previously
    /// passed `&str` to `execute_root_turn`).
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![ContentBlock::Text { text: text.into() }],
            audit_override: None,
        }
    }

    /// Construct from a pre-built block list.
    #[must_use]
    pub const fn from_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            blocks,
            audit_override: None,
        }
    }

    /// Synthetic input for a resume retry. The `blocks` list is empty
    /// (the resume path doesn't append a user message — the staged
    /// store already contains everything the LLM needs) and
    /// `audit_summary()` returns [`RESUME_AUDIT_PROMPT`].
    #[must_use]
    pub const fn resume() -> Self {
        Self {
            blocks: Vec::new(),
            audit_override: Some(RESUME_AUDIT_PROMPT),
        }
    }

    /// Whether this input came from [`Self::resume`] — used by the
    /// retry / compaction-recovery paths to decide whether to push a
    /// fresh user message into the rebuilt chat request.
    #[must_use]
    pub const fn is_resume(&self) -> bool {
        self.audit_override.is_some()
    }

    /// Whether the input has zero content blocks. True for resume
    /// inputs and any caller-constructed empty input. The worker
    /// rejects empty fresh-turn inputs upstream.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// View the underlying blocks. Used by `build_chat_request` and
    /// `buffer_turn_messages` to construct the user `Message`.
    #[must_use]
    pub fn blocks(&self) -> &[ContentBlock] {
        &self.blocks
    }

    /// Consume into an `llm::Message` with role `User`. Returns
    /// `None` for resume inputs (they have no message to push).
    ///
    /// Uses `Message::user_with_content` even for single-text inputs
    /// so the wire format is uniform across the worker — both the
    /// staged store and the chat request see `Content::Blocks(...)`
    /// regardless of whether the input was text-only or mixed.
    #[must_use]
    pub fn into_message(self) -> Option<Message> {
        if self.is_resume() {
            None
        } else {
            Some(Message::user_with_content(self.blocks))
        }
    }

    /// Audit-string projection: `"<resume>"` for resume inputs,
    /// otherwise text-block content joined by newlines with binary
    /// attachments replaced by `[<media_type> attachment]`.
    ///
    /// Stored in `TurnAttempt::request_blob.user_prompt` and used by
    /// `LlmRetryParams::attempt_audit_prompt` for resume detection.
    #[must_use]
    pub fn audit_summary(&self) -> String {
        if let Some(sentinel) = self.audit_override {
            return sentinel.to_owned();
        }
        let parts: Vec<String> = self
            .blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                ContentBlock::Image { source } | ContentBlock::Document { source } => {
                    Some(format!("[{} attachment]", source.media_type))
                }
                // Non-text/attachment blocks contribute nothing to the audit
                // summary, and the `_` arm covers future `#[non_exhaustive]`
                // block kinds.
                _ => None,
            })
            .collect();
        parts.join("\n")
    }
}

impl Default for UserInput {
    fn default() -> Self {
        Self::from_blocks(Vec::new())
    }
}

impl From<&str> for UserInput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<&String> for UserInput {
    fn from(text: &String) -> Self {
        Self::text(text.clone())
    }
}

impl From<String> for UserInput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<Vec<ContentBlock>> for UserInput {
    fn from(blocks: Vec<ContentBlock>) -> Self {
        Self::from_blocks(blocks)
    }
}

/// Build a [`UserInput`] from the durable `task.submitted_input` vector.
///
/// Used by `agent-service-host`'s root-task entry point to flow gRPC
/// `UserInputItem` payloads (text + image + document) into the worker
/// without lossy string flattening.
///
/// Maps:
///
/// * `SubmittedInputItem::Text { text }` → `ContentBlock::Text { text }`
/// * `SubmittedInputItem::Image { media_type, data_base64 }` →
///   `ContentBlock::Image { source: ContentSource { media_type, data: data_base64 } }`
/// * `SubmittedInputItem::Document { media_type, data_base64 }` →
///   `ContentBlock::Document { source: ContentSource { media_type, data: data_base64 } }`
///
/// `data_base64` flows through unchanged because the providers'
/// wire-format converters (e.g.
/// `agent-sdk-providers::impls::anthropic::ApiSource::from_content_source`)
/// already expect base64-encoded strings.
#[must_use]
pub fn user_input_from_submitted(items: &[SubmittedInputItem]) -> UserInput {
    let blocks = items
        .iter()
        .map(|item| match item {
            SubmittedInputItem::Text { text } => ContentBlock::Text { text: text.clone() },
            SubmittedInputItem::Image {
                media_type,
                data_base64,
            } => ContentBlock::Image {
                source: ContentSource::new(media_type.clone(), data_base64.clone()),
            },
            SubmittedInputItem::Document {
                media_type,
                data_base64,
            } => ContentBlock::Document {
                source: ContentSource::new(media_type.clone(), data_base64.clone()),
            },
        })
        .collect();
    UserInput::from_blocks(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_input_round_trips_to_message() {
        let input = UserInput::text("hello");
        assert!(!input.is_resume());
        assert_eq!(input.audit_summary(), "hello");
        let msg = input.into_message().expect("text input has a message");
        assert!(matches!(
            msg.content,
            agent_sdk_foundation::llm::Content::Blocks(_)
        ));
    }

    #[test]
    fn resume_input_has_no_message() {
        let input = UserInput::resume();
        assert!(input.is_resume());
        assert_eq!(input.audit_summary(), RESUME_AUDIT_PROMPT);
        assert!(input.into_message().is_none());
    }

    #[test]
    fn audit_summary_describes_attachments() {
        let input = UserInput::from_blocks(vec![
            ContentBlock::Text {
                text: "what is this".into(),
            },
            ContentBlock::Image {
                source: ContentSource::new("image/png", "AAAA"),
            },
            ContentBlock::Document {
                source: ContentSource::new("application/pdf", "BBBB"),
            },
        ]);
        assert_eq!(
            input.audit_summary(),
            "what is this\n[image/png attachment]\n[application/pdf attachment]",
        );
    }

    #[test]
    fn from_submitted_handles_every_variant() {
        let items = vec![
            SubmittedInputItem::Text { text: "hi".into() },
            SubmittedInputItem::Image {
                media_type: "image/jpeg".into(),
                data_base64: "AAAA".into(),
            },
            SubmittedInputItem::Document {
                media_type: "text/plain".into(),
                data_base64: "BBBB".into(),
            },
        ];
        let input = user_input_from_submitted(&items);
        assert_eq!(input.blocks().len(), 3);
        assert!(matches!(&input.blocks()[0], ContentBlock::Text { .. }));
        assert!(matches!(
            &input.blocks()[1],
            ContentBlock::Image { source } if source.media_type == "image/jpeg" && source.data == "AAAA"
        ));
        assert!(matches!(
            &input.blocks()[2],
            ContentBlock::Document { source } if source.media_type == "text/plain" && source.data == "BBBB"
        ));
    }

    #[test]
    fn from_str_and_string_yield_text() {
        let from_str: UserInput = "hello".into();
        let from_string: UserInput = String::from("hello").into();
        assert_eq!(from_str.audit_summary(), "hello");
        assert_eq!(from_string.audit_summary(), "hello");
    }
}
