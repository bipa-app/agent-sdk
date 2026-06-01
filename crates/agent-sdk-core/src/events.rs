//! Agent events for real-time streaming.
//!
//! The [`AgentEvent`] enum represents all events that can occur during agent
//! execution. These events are streamed via an async channel for real-time
//! UI updates and logging.
//!
//! # Event Flow
//!
//! A typical event sequence looks like:
//! 1. `Start` - Agent begins processing
//! 2. `Text` / `ToolCallStart` / `ToolCallEnd` - Processing events
//! 3. `TurnComplete` - One LLM round-trip finished
//! 4. `Done` - Agent completed successfully, or `Error` if failed

use crate::llm::ContentBlock;
use crate::types::{ThreadId, TokenUsage, ToolResult, ToolTier};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use time::OffsetDateTime;

/// Events emitted by the agent loop during execution.
/// These are streamed to the client for real-time UI updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentEvent {
    /// Agent loop has started
    Start { thread_id: ThreadId, turn: usize },

    /// The user prompt that opens a turn.
    ///
    /// Committed by the worker on the **first attempt** of each
    /// root-turn task, immediately before the matching
    /// [`AgentEvent::Start`]. Carries the task's admitted
    /// `submitted_input` lifted into the LLM-content shape
    /// (`Vec<ContentBlock>` — text, image, document) so consumers
    /// can render the prompt without reaching into the projection
    /// or the task store. Retries of the same turn do not re-emit
    /// the event; downstream readers can pair `UserInput` 1:1
    /// with the *first* `Start { turn: N }` per turn.
    ///
    /// This is the durable, sequence-numbered admission event the
    /// projection never carried — `MessageProjection::messages`
    /// still holds the same prompt as an `llm::Message`, but it
    /// has no sequence and commingles with tool-result and
    /// compaction-summary user-role rows. Replay clients that
    /// need a clean, chronological "this is what the user typed"
    /// signal read this event instead.
    UserInput {
        thread_id: ThreadId,
        /// Lifted from the admitted task's
        /// `submitted_input`. Only `Text`, `Image`, and `Document`
        /// blocks appear — the runtime never admits user prompts
        /// containing tool blocks, but the broader
        /// `ContentBlock` type lets the field round-trip through
        /// the same wire shapes the projection uses.
        content: Vec<ContentBlock>,
    },

    /// Agent is "thinking" - complete thinking text after stream ends
    Thinking { message_id: String, text: String },

    /// A thinking delta for streaming thinking content
    ThinkingDelta { message_id: String, delta: String },

    /// A text delta for streaming responses
    TextDelta { message_id: String, delta: String },

    /// Complete text block from the agent
    Text { message_id: String, text: String },

    /// Agent is about to call a tool
    ToolCallStart {
        id: String,
        name: String,
        display_name: String,
        input: serde_json::Value,
        tier: ToolTier,
    },

    /// Tool execution completed
    ToolCallEnd {
        id: String,
        name: String,
        display_name: String,
        result: ToolResult,
    },

    /// Progress update from an async tool operation
    ToolProgress {
        /// Tool call ID
        id: String,
        /// Tool name
        name: String,
        /// Human-readable display name
        display_name: String,
        /// Progress stage
        stage: String,
        /// Human-readable progress message
        message: String,
        /// Optional tool-specific data
        data: Option<serde_json::Value>,
    },

    /// Tool requires confirmation before execution.
    /// The application determines the confirmation type (normal, PIN, biometric).
    ToolRequiresConfirmation {
        id: String,
        name: String,
        display_name: String,
        input: serde_json::Value,
        description: String,
    },

    /// Agent turn completed (one LLM round-trip)
    TurnComplete { turn: usize, usage: TokenUsage },

    /// Agent loop completed successfully
    Done {
        thread_id: ThreadId,
        total_turns: usize,
        total_usage: TokenUsage,
        duration: Duration,
    },

    /// An error occurred during execution
    Error { message: String, recoverable: bool },

    /// Auto-retry was initiated for a transient LLM error (rate
    /// limit, server error, etc.). The `delay_ms` field gives the
    /// runtime's chosen backoff before re-attempting; consumers
    /// can render a "Retrying X/N in Ys…" indicator and clear it
    /// on the matching `AutoRetryEnd`.
    AutoRetryStart {
        /// 1-based retry attempt number (first retry = 1).
        attempt: u32,
        /// Maximum retry attempts configured for this run.
        max_attempts: u32,
        /// Backoff before the next attempt in milliseconds.
        delay_ms: u64,
        /// Human-readable reason the retry was triggered.
        error_message: String,
    },

    /// Auto-retry settled. `success = true` means a subsequent
    /// attempt succeeded; `success = false` means the retry budget
    /// was exhausted and `final_error` carries the last error.
    AutoRetryEnd {
        /// Total attempts performed (matches the last
        /// `AutoRetryStart`'s `attempt`).
        attempt: u32,
        /// Whether a follow-up attempt eventually succeeded.
        success: bool,
        /// Last error when the retry budget ran out.
        final_error: Option<String>,
    },

    /// The model refused the request (safety/policy).
    Refusal {
        message_id: String,
        text: Option<String>,
    },

    /// The run was cancelled via its [`CancellationToken`].
    ///
    /// This is a **terminal** event, emitted exactly once on every
    /// cancellation return site (mirroring [`AgentEvent::Done`] and
    /// [`AgentEvent::Refusal`]). Cancellation can land at the top of a
    /// turn, mid-stream while the model is still producing tokens,
    /// while a tool is in flight, or during context compaction — in
    /// every case the run closes with this event so a streaming
    /// consumer receives a closing marker and never hangs waiting for
    /// `Done`.
    ///
    /// `turn` is the turn number reached when the cancel was honored
    /// and `usage` is the partial token usage accumulated so far.
    ///
    /// [`CancellationToken`]: https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html
    Cancelled { turn: usize, usage: TokenUsage },

    /// Context was compacted to reduce size
    ContextCompacted {
        /// Number of messages before compaction
        original_count: usize,
        /// Number of messages after compaction
        new_count: usize,
        /// Estimated tokens before compaction
        original_tokens: usize,
        /// Estimated tokens after compaction
        new_tokens: usize,
    },

    /// Progress update from a running subagent
    SubagentProgress {
        /// ID of the parent tool call that spawned this subagent
        subagent_id: String,
        /// Name of the subagent (e.g., "explore", "plan")
        subagent_name: String,
        /// Human-friendly nickname assigned by the parent (e.g., "Zara")
        nickname: Option<String>,
        /// Durable child thread reference, when available.
        child_thread_id: Option<ThreadId>,
        /// Durable child root task reference, when available.
        child_root_task_id: Option<String>,
        /// Durable parent-visible invocation task reference, when available.
        subagent_task_id: Option<String>,
        /// Maximum turns configured for this subagent
        max_turns: Option<u32>,
        /// Current turn number of the subagent
        current_turn: Option<u32>,
        /// Model being used by the subagent
        model: Option<String>,
        /// Summary label associated with the latest subagent update.
        tool_name: String,
        /// Brief context associated with the latest subagent update.
        tool_context: String,
        /// Whether the summarized update represents terminal completion.
        completed: bool,
        /// Whether the subagent succeeded (only meaningful if completed)
        success: bool,
        /// Current total tool count for this subagent
        tool_count: u32,
        /// Current total tokens used by this subagent
        total_tokens: u64,
    },
}

impl AgentEvent {
    #[must_use]
    pub const fn start(thread_id: ThreadId, turn: usize) -> Self {
        Self::Start { thread_id, turn }
    }

    #[must_use]
    pub const fn user_input(thread_id: ThreadId, content: Vec<ContentBlock>) -> Self {
        Self::UserInput { thread_id, content }
    }

    #[must_use]
    pub fn thinking(message_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Thinking {
            message_id: message_id.into(),
            text: text.into(),
        }
    }

    #[must_use]
    pub fn thinking_delta(message_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self::ThinkingDelta {
            message_id: message_id.into(),
            delta: delta.into(),
        }
    }

    #[must_use]
    pub fn text_delta(message_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self::TextDelta {
            message_id: message_id.into(),
            delta: delta.into(),
        }
    }

    #[must_use]
    pub fn text(message_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Text {
            message_id: message_id.into(),
            text: text.into(),
        }
    }

    #[must_use]
    pub fn tool_call_start(
        id: impl Into<String>,
        name: impl Into<String>,
        display_name: impl Into<String>,
        input: serde_json::Value,
        tier: ToolTier,
    ) -> Self {
        Self::ToolCallStart {
            id: id.into(),
            name: name.into(),
            display_name: display_name.into(),
            input,
            tier,
        }
    }

    #[must_use]
    pub fn tool_call_end(
        id: impl Into<String>,
        name: impl Into<String>,
        display_name: impl Into<String>,
        result: ToolResult,
    ) -> Self {
        Self::ToolCallEnd {
            id: id.into(),
            name: name.into(),
            display_name: display_name.into(),
            result,
        }
    }

    #[must_use]
    pub fn tool_progress(
        id: impl Into<String>,
        name: impl Into<String>,
        display_name: impl Into<String>,
        stage: impl Into<String>,
        message: impl Into<String>,
        data: Option<serde_json::Value>,
    ) -> Self {
        Self::ToolProgress {
            id: id.into(),
            name: name.into(),
            display_name: display_name.into(),
            stage: stage.into(),
            message: message.into(),
            data,
        }
    }

    #[must_use]
    pub fn tool_requires_confirmation(
        id: impl Into<String>,
        name: impl Into<String>,
        display_name: impl Into<String>,
        input: serde_json::Value,
        description: impl Into<String>,
    ) -> Self {
        Self::ToolRequiresConfirmation {
            id: id.into(),
            name: name.into(),
            display_name: display_name.into(),
            input,
            description: description.into(),
        }
    }

    #[must_use]
    pub const fn done(
        thread_id: ThreadId,
        total_turns: usize,
        total_usage: TokenUsage,
        duration: Duration,
    ) -> Self {
        Self::Done {
            thread_id,
            total_turns,
            total_usage,
            duration,
        }
    }

    #[must_use]
    pub fn error(message: impl Into<String>, recoverable: bool) -> Self {
        Self::Error {
            message: message.into(),
            recoverable,
        }
    }

    #[must_use]
    pub fn refusal(message_id: impl Into<String>, text: Option<String>) -> Self {
        Self::Refusal {
            message_id: message_id.into(),
            text,
        }
    }

    #[must_use]
    pub const fn cancelled(turn: usize, usage: TokenUsage) -> Self {
        Self::Cancelled { turn, usage }
    }

    #[must_use]
    pub const fn context_compacted(
        original_count: usize,
        new_count: usize,
        original_tokens: usize,
        new_tokens: usize,
    ) -> Self {
        Self::ContextCompacted {
            original_count,
            new_count,
            original_tokens,
            new_tokens,
        }
    }
}

/// Monotonically increasing per-run counter for event ordering.
///
/// Each `run()` or `run_turn()` call creates a fresh counter starting at 0.
/// The counter is `Arc`-wrapped so it can be shared across tasks (e.g., subagent
/// progress events sent from child tokio tasks).
///
/// `Ordering::Relaxed` is sufficient because the mpsc channel provides the
/// happens-before ordering guarantee between sender and receiver.
#[derive(Clone, Debug)]
pub struct SequenceCounter(Arc<AtomicU64>);

impl SequenceCounter {
    /// Create a new counter starting at 0.
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }

    /// Create a counter starting at the given offset.
    ///
    /// Used by server mode to resume sequencing across turns within
    /// the same thread — the server seeds the counter with the last
    /// known sequence value so numbering is continuous.
    #[must_use]
    pub fn with_offset(start: u64) -> Self {
        Self(Arc::new(AtomicU64::new(start)))
    }

    /// Get the next sequence number, incrementing the counter.
    #[must_use]
    pub fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for SequenceCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Envelope wrapping every [`AgentEvent`] with idempotency metadata.
///
/// Mobile clients can use `event_id` for deduplication on retry, `sequence`
/// for ordering after persistence, and `timestamp` for display.
///
/// The `event` field is flattened in JSON so that `event_id`, `sequence`,
/// `timestamp`, and the event's `type` discriminant all appear at the same level.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentEventEnvelope {
    /// Unique identifier for this event emission.
    ///
    /// UUID v4 when created via [`AgentEventEnvelope::wrap`] (SDK-local path),
    /// UUID v7 when created via server-committed `CommittedEvent::into_envelope`.
    pub event_id: uuid::Uuid,
    /// Monotonically increasing sequence number within a single run.
    pub sequence: u64,
    /// UTC timestamp of when the event was emitted.
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: OffsetDateTime,
    /// The actual event payload.
    #[serde(flatten)]
    pub event: AgentEvent,
}

impl AgentEventEnvelope {
    /// Wrap an [`AgentEvent`] in an envelope, assigning it a unique ID,
    /// the next sequence number, and the current UTC timestamp.
    #[must_use]
    pub fn wrap(event: AgentEvent, seq: &SequenceCounter) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4(),
            sequence: seq.next(),
            timestamp: OffsetDateTime::now_utc(),
            event,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ===================
    // SequenceCounter
    // ===================

    #[test]
    fn sequence_counter_starts_at_zero() {
        let seq = SequenceCounter::new();
        assert_eq!(seq.next(), 0);
    }

    #[test]
    fn sequence_counter_increments_monotonically() {
        let seq = SequenceCounter::new();
        for expected in 0..100 {
            assert_eq!(seq.next(), expected);
        }
    }

    #[test]
    fn sequence_counter_no_gaps() {
        let seq = SequenceCounter::new();
        let values: Vec<u64> = (0..50).map(|_| seq.next()).collect();
        let expected: Vec<u64> = (0..50).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn sequence_counter_clones_share_state() {
        let seq = SequenceCounter::new();
        let clone = seq.clone();

        assert_eq!(seq.next(), 0);
        assert_eq!(clone.next(), 1);
        assert_eq!(seq.next(), 2);
    }

    #[test]
    fn sequence_counter_default_starts_at_zero() {
        let seq = SequenceCounter::default();
        assert_eq!(seq.next(), 0);
    }

    #[test]
    fn sequence_counter_with_offset_starts_at_given_value() {
        let seq = SequenceCounter::with_offset(42);
        assert_eq!(seq.next(), 42);
        assert_eq!(seq.next(), 43);
        assert_eq!(seq.next(), 44);
    }

    #[test]
    fn sequence_counter_with_offset_zero_same_as_new() {
        let seq = SequenceCounter::with_offset(0);
        assert_eq!(seq.next(), 0);
        assert_eq!(seq.next(), 1);
    }

    #[tokio::test]
    async fn sequence_counter_unique_across_concurrent_tasks() {
        let seq = SequenceCounter::new();
        let n = 1000;

        let mut handles = Vec::new();
        for _ in 0..n {
            let seq_clone = seq.clone();
            handles.push(tokio::spawn(async move { seq_clone.next() }));
        }

        let mut values = HashSet::new();
        for handle in handles {
            let val = handle.await.unwrap();
            assert!(values.insert(val), "duplicate sequence number: {val}");
        }

        assert_eq!(values.len(), n);
        // All values should be in [0, n)
        for v in &values {
            assert!(*v < n as u64);
        }
    }

    // ===================
    // AgentEventEnvelope
    // ===================

    fn sample_event() -> AgentEvent {
        AgentEvent::text("msg_1", "hello")
    }

    #[test]
    fn wrap_assigns_unique_event_ids() {
        let seq = SequenceCounter::new();
        let ids: HashSet<uuid::Uuid> = (0..100)
            .map(|_| AgentEventEnvelope::wrap(sample_event(), &seq).event_id)
            .collect();
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn wrap_event_id_is_valid_uuid_v4() {
        let seq = SequenceCounter::new();
        let envelope = AgentEventEnvelope::wrap(sample_event(), &seq);
        assert_eq!(envelope.event_id.get_version(), Some(uuid::Version::Random));
    }

    #[test]
    fn wrap_assigns_incrementing_sequences() {
        let seq = SequenceCounter::new();
        let envelopes: Vec<AgentEventEnvelope> = (0..10)
            .map(|_| AgentEventEnvelope::wrap(sample_event(), &seq))
            .collect();

        for (i, env) in envelopes.iter().enumerate() {
            assert_eq!(env.sequence, i as u64);
        }
    }

    #[test]
    fn wrap_timestamps_are_non_decreasing() {
        let seq = SequenceCounter::new();
        let envelopes: Vec<AgentEventEnvelope> = (0..20)
            .map(|_| AgentEventEnvelope::wrap(sample_event(), &seq))
            .collect();

        for pair in envelopes.windows(2) {
            assert!(pair[1].timestamp >= pair[0].timestamp);
        }
    }

    #[test]
    fn wrap_preserves_inner_event() {
        let seq = SequenceCounter::new();
        let envelope = AgentEventEnvelope::wrap(AgentEvent::text("msg_42", "content"), &seq);
        match &envelope.event {
            AgentEvent::Text { message_id, text } => {
                assert_eq!(message_id, "msg_42");
                assert_eq!(text, "content");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn separate_counters_produce_independent_sequences() {
        let seq_a = SequenceCounter::new();
        let seq_b = SequenceCounter::new();

        let a0 = AgentEventEnvelope::wrap(sample_event(), &seq_a);
        let b0 = AgentEventEnvelope::wrap(sample_event(), &seq_b);
        let a1 = AgentEventEnvelope::wrap(sample_event(), &seq_a);
        let b1 = AgentEventEnvelope::wrap(sample_event(), &seq_b);

        // Both start at 0 independently
        assert_eq!(a0.sequence, 0);
        assert_eq!(b0.sequence, 0);
        assert_eq!(a1.sequence, 1);
        assert_eq!(b1.sequence, 1);

        // But event_ids are still globally unique
        let ids: HashSet<uuid::Uuid> = [&a0, &b0, &a1, &b1].iter().map(|e| e.event_id).collect();
        assert_eq!(ids.len(), 4);
    }

    // ===================
    // Serialization
    // ===================

    #[test]
    fn envelope_serializes_flat_json() {
        let seq = SequenceCounter::new();
        let envelope = AgentEventEnvelope::wrap(AgentEvent::text("msg_1", "hi"), &seq);
        let json: serde_json::Value = serde_json::to_value(&envelope).expect("serialize");

        // Top-level fields from the envelope
        assert!(json.get("event_id").is_some());
        assert!(json.get("sequence").is_some());
        assert!(json.get("timestamp").is_some());

        // Flattened event fields at the same level
        assert_eq!(json.get("type").and_then(|v| v.as_str()), Some("text"));
        assert_eq!(
            json.get("message_id").and_then(|v| v.as_str()),
            Some("msg_1")
        );
        assert_eq!(json.get("text").and_then(|v| v.as_str()), Some("hi"));

        // No nested "event" key
        assert!(json.get("event").is_none());
    }

    #[test]
    fn envelope_event_id_does_not_collide_with_tool_id() {
        let seq = SequenceCounter::new();
        let envelope = AgentEventEnvelope::wrap(
            AgentEvent::tool_call_start(
                "tool_123",
                "bash",
                "Bash",
                serde_json::json!({}),
                ToolTier::Observe,
            ),
            &seq,
        );
        let json: serde_json::Value = serde_json::to_value(&envelope).expect("serialize");

        // Both `event_id` and tool `id` are present and distinct
        let event_id = json.get("event_id").and_then(|v| v.as_str()).unwrap();
        let tool_id = json.get("id").and_then(|v| v.as_str()).unwrap();
        assert_ne!(event_id, tool_id);
        assert_eq!(tool_id, "tool_123");
    }

    #[test]
    fn envelope_roundtrip_serde() {
        let seq = SequenceCounter::new();
        let original = AgentEventEnvelope::wrap(AgentEvent::text("msg_1", "hello"), &seq);

        let json_str = serde_json::to_string(&original).expect("serialize");
        let restored: AgentEventEnvelope = serde_json::from_str(&json_str).expect("deserialize");

        assert_eq!(restored.event_id, original.event_id);
        assert_eq!(restored.sequence, original.sequence);
        assert_eq!(restored.timestamp, original.timestamp);
        match &restored.event {
            AgentEvent::Text { message_id, text } => {
                assert_eq!(message_id, "msg_1");
                assert_eq!(text, "hello");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn envelope_sequence_is_u64_in_json() {
        let seq = SequenceCounter::new();
        let envelope = AgentEventEnvelope::wrap(sample_event(), &seq);
        let json: serde_json::Value = serde_json::to_value(&envelope).expect("serialize");

        assert!(json.get("sequence").unwrap().is_u64());
        assert_eq!(json.get("sequence").unwrap().as_u64(), Some(0));
    }

    #[test]
    fn envelope_timestamp_is_rfc3339_string() {
        let seq = SequenceCounter::new();
        let envelope = AgentEventEnvelope::wrap(sample_event(), &seq);
        let json: serde_json::Value = serde_json::to_value(&envelope).expect("serialize");

        let ts_str = json.get("timestamp").unwrap().as_str().unwrap();
        // Should parse as RFC 3339
        time::OffsetDateTime::parse(ts_str, &time::format_description::well_known::Rfc3339)
            .expect("timestamp should be valid RFC 3339");
    }
}
