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
use crate::types::{BudgetLimitKind, ThreadId, TokenUsage, ToolResult, ToolTier};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use time::OffsetDateTime;

/// Serde adapter encoding a [`Duration`] as a millisecond integer
/// (`duration_ms`) instead of serde's default `{secs,nanos}` object.
///
/// Deserialization is deliberately lenient: durable event rows written
/// before the wire form changed to `duration_ms` carry serde's default
/// `{"secs":..,"nanos":..}` object (under the old `duration` key, accepted
/// via `#[serde(alias = "duration")]` on the fields). Hosts replay those
/// rows with `serde_json::from_value`, so both representations must decode
/// or every thread containing a pre-change terminal event becomes
/// unreadable after an upgrade. Serialization always writes millis.
mod duration_ms_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    /// The two wire shapes a duration value can arrive in.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DurationRepr {
        /// Current form: a flat millisecond integer.
        Millis(u64),
        /// Legacy form: serde's default `Duration` object.
        Legacy { secs: u64, nanos: u32 },
    }

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
        serializer.serialize_u64(ms)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        match DurationRepr::deserialize(deserializer)? {
            DurationRepr::Millis(ms) => Ok(Duration::from_millis(ms)),
            DurationRepr::Legacy { secs, nanos } => Ok(Duration::new(secs, nanos)),
        }
    }
}

/// Typed explanation for a durable task reaching a terminal state.
///
/// The tagged wire shape keeps variants stable while allowing structured
/// details such as the provider error `kind`. `Unknown` is the explicit
/// forward-compatibility sink for a newer producer's reason; callers must still
/// name it in exhaustive matches rather than silently discarding it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum TerminalReason {
    /// The task completed successfully.
    Completed,
    /// The caller explicitly cancelled this task tree.
    UserCancel,
    /// A configured turn, token, cost, or retry budget was exhausted.
    Budget,
    /// The subagent watchdog observed no work for the full stall budget.
    WatchdogStall,
    /// The provider failed the request with a provider-specific category.
    ProviderError { kind: String },
    /// An ancestor was cancelled, so this descendant was cancelled with it.
    ParentCancelled,
    /// A pending confirmation was rejected, revoked, or timed out.
    ConfirmationRejected,
    /// An internal execution or persistence error terminated the task.
    InternalError,
    /// A reason introduced by a newer producer.
    #[serde(other)]
    Unknown,
}

/// What started a context compaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionTrigger {
    /// The pre-call occupancy check crossed the threshold.
    Threshold,
    /// The provider rejected a request for exceeding its context window.
    Overflow,
    /// A user or host explicitly asked for compaction.
    UserRequested,
}

/// Compaction implementation selected by the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionEngine {
    /// Summarize the selected prefix without pruning and use estimated triggers.
    #[default]
    Legacy,
    /// Prune recoverable tool output first and let the host use measured triggers.
    PruneFirst,
    /// Deterministic local transcript rendering with an exact source archive.
    Snapcompact,
}

/// How the occupancy that crossed a compaction threshold was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccupancySource {
    /// The provider's input-token count for the latest request with the
    /// same system prompt and tools, plus an estimate of what was appended
    /// since.
    Measured,
    /// A character-count estimate of the history alone.
    Estimated,
    /// A source introduced by a newer producer.
    #[serde(other)]
    Unknown,
}

/// Where a compaction threshold came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdSource {
    /// The serving model's context window less the output reserve.
    ModelAware,
    /// The host's fixed threshold; the host did not ask for a model-aware one.
    Fixed,
    /// The host's fixed threshold, because the serving model's context
    /// window is unknown.
    FixedUnknownModel,
    /// The host's fixed threshold, because the output reserve fills the
    /// serving model's whole context window.
    FixedInvalidModelWindow,
    /// A source introduced by a newer producer.
    #[serde(other)]
    Unknown,
}

impl ThresholdSource {
    /// Stable `snake_case` name, as serialized.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelAware => "model_aware",
            Self::Fixed => "fixed",
            Self::FixedUnknownModel => "fixed_unknown_model",
            Self::FixedInvalidModelWindow => "fixed_invalid_model_window",
            Self::Unknown => "unknown",
        }
    }
}

/// The request occupancy that crossed a compaction threshold: the system
/// prompt, tools and history of the request about to be sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionOccupancy {
    pub tokens: usize,
    pub source: OccupancySource,
}

/// The threshold a compaction trigger compared the occupancy against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionThreshold {
    pub tokens: usize,
    pub source: ThresholdSource,
    /// Context window the threshold was derived from; `None` when it is
    /// unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window_tokens: Option<usize>,
}

/// Decodes an optional event field, mapping values this build does not know
/// to `None` so journal rows from newer producers stay readable.
fn known_or_none<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| serde_json::from_value(value).ok()))
}

/// What a running subagent is doing, as its live progress frame reports it.
///
/// The unknown sink keeps durable journal rows readable when a newer producer
/// introduces another phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentPhase {
    /// The child's own turn is running.
    Working,
    /// The child's turn is parked on its tool calls.
    WaitingOnTools,
    /// The child has shown no sign of work for at least half its stall
    /// budget. Reported only; nothing is cancelled.
    Stalled,
    /// A phase introduced by a newer producer.
    #[serde(other)]
    Unknown,
}

impl SubagentPhase {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::WaitingOnTools => "waiting_on_tools",
            Self::Stalled => "stalled",
            Self::Unknown => "unknown",
        }
    }
}

/// The live part of a periodic [`AgentEvent::SubagentProgress`] frame.
///
/// Only the frames the worker writes while the child runs carry it; the
/// spawn and completion frames do not.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentLiveness {
    /// The newest sign of work in the child: its live activity beacon, the
    /// durable `last_activity_at` of its thread's tasks, or its newest LLM
    /// call.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub last_activity_at: Option<OffsetDateTime>,
    pub phase: SubagentPhase,
    /// LLM calls the child's root task has made, including the open one.
    pub llm_calls: u64,
    /// Output tokens over the child's finished LLM calls, including turns
    /// that have not committed yet.
    pub output_tokens_live: u64,
}

/// Why a daemon changed the OAuth account serving a provider request.
///
/// The unknown sink keeps durable journal rows readable when a newer producer
/// introduces another reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AccountRotationReason {
    /// The provider returned a typed rate-limit response.
    RateLimited,
    /// The provider rejected the active account after refresh was exhausted.
    AuthenticationFailed,
    /// The active account could not refresh before the request was dispatched.
    RefreshFailed,
    /// A reason introduced by a newer producer.
    #[serde(other)]
    Unknown,
}

/// Events emitted by the agent loop during execution.
/// These are streamed to the client for real-time UI updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentEvent {
    /// A thread aggregate became durably visible.
    ThreadCreated {
        thread_id: ThreadId,
        /// Source thread when this thread was created by `ForkThread`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_thread_id: Option<ThreadId>,
        /// Source turn boundary copied by `ForkThread`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fork_after_committed_turns: Option<u32>,
    },

    /// Agent loop has started
    Start {
        thread_id: ThreadId,
        turn: usize,
        /// Durable task that committed this event. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// The user prompt that opens a turn, or input injected into a
    /// running turn.
    ///
    /// A turn's prompt is committed **once per root-turn task**,
    /// immediately before that task's first [`AgentEvent::Start`]; a
    /// re-acquired task never commits it again, so readers can pair
    /// each prompt 1:1 with the *first* `Start { turn: N }` per turn.
    /// Input injected into a running turn is committed when the turn
    /// drains it at a tool boundary, so it follows that turn's `Start`.
    /// Carries the admitted `submitted_input` lifted into the
    /// LLM-content shape (`Vec<ContentBlock>` — text, image, document)
    /// so consumers can render it without reaching into the projection
    /// or the task store.
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
        /// Durable task that admitted this input: the root-turn task for
        /// a turn's prompt, the input-injection task for injected input.
        /// Absent on rows journaled before prompts were stamped and on
        /// runs without a durable task. Its presence does not mark an
        /// injection: injected input names a different task than the
        /// `emitter_task_id` of the turn's `Start`, a prompt names the same.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// Agent is "thinking" - complete thinking text after stream ends
    Thinking {
        message_id: String,
        text: String,
        /// Durable task whose execution produced this content. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// A thinking delta for streaming thinking content
    ThinkingDelta {
        message_id: String,
        delta: String,
        /// Durable task whose execution produced this content. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// A text delta for streaming responses
    TextDelta {
        message_id: String,
        delta: String,
        /// Durable task whose execution produced this content. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// Complete text block from the agent
    Text {
        message_id: String,
        text: String,
        /// Durable task whose execution produced this content. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

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

    /// Agent execution is durably parked until the user answers every
    /// question in the batch.
    QuestionAsked {
        /// Root task to pass to the `AnswerQuestion` RPC.
        task_id: String,
        /// User-facing questions, one per pending `ask_user` call, in
        /// pending-tool-call order. The `AnswerQuestion` RPC must
        /// resolve all of them in one call.
        questions: Vec<crate::QuestionPayload>,
    },

    /// Agent turn completed (one LLM round-trip)
    TurnComplete {
        turn: usize,
        usage: TokenUsage,
        /// Durable task that committed this event. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// Agent loop completed successfully
    Done {
        thread_id: ThreadId,
        total_turns: usize,
        total_usage: TokenUsage,
        /// Wall-clock run duration.
        ///
        /// Serialized on the wire as `duration_ms` (a millisecond integer) to
        /// match [`TurnSummary::duration_ms`](crate::types::TurnSummary) — the
        /// flattened envelope previously encoded this as a nested
        /// `{"secs":..,"nanos":..}` object, inconsistent with the rest of the
        /// streaming contract. The Rust field keeps the `Duration` type.
        #[serde(rename = "duration_ms", alias = "duration", with = "duration_ms_serde")]
        duration: Duration,
        /// Estimated cost of the run in USD, when the run's provider/model
        /// has pricing metadata. Omitted from the wire form when `None` so
        /// the streaming contract stays compatible with consumers that
        /// predate cost accounting.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        estimated_cost_usd: Option<f64>,
        /// Durable task that committed this event. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// The run was stopped because a run-level usage budget was exceeded.
    ///
    /// This is a **terminal** event, emitted once on the budget-exceeded
    /// return site in place of [`AgentEvent::Done`], so a streaming
    /// consumer always receives a closing marker. `limit` identifies which
    /// budget tripped.
    BudgetExceeded {
        thread_id: ThreadId,
        total_turns: usize,
        total_usage: TokenUsage,
        /// Wall-clock run duration up to the moment the budget tripped.
        ///
        /// Serialized on the wire as `duration_ms` (a millisecond integer),
        /// mirroring [`AgentEvent::Done`].
        #[serde(rename = "duration_ms", alias = "duration", with = "duration_ms_serde")]
        duration: Duration,
        /// Estimated cost of the run in USD at the moment the budget was
        /// hit, when pricing metadata is available.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        estimated_cost_usd: Option<f64>,
        /// Which budget limit was exceeded.
        limit: BudgetLimitKind,
        /// Durable task that committed this event. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// An error occurred during execution
    Error {
        message: String,
        recoverable: bool,
        /// Typed terminal explanation when this error closes a durable task.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<TerminalReason>,
        /// Durable task that committed this event. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

    /// Auto-retry was initiated for a recoverable LLM error (rate
    /// limit, server error, connectivity loss). The `delay_ms` field
    /// gives the runtime's chosen backoff before re-attempting;
    /// consumers can render a "Retrying X/N in Ys…" indicator and
    /// clear it on the matching `AutoRetryEnd`.
    AutoRetryStart {
        /// 1-based failure ordinal within the turn (first failure = 1).
        ///
        /// Not necessarily contiguous across events: a connectivity
        /// streak emits one `AutoRetryStart` (on its first failure)
        /// while later failures in the streak still consume ordinals,
        /// so consecutive events can read e.g. `1, 2, 4`.
        attempt: u32,
        /// Maximum retry attempts configured for this run.
        ///
        /// `u32::MAX` is a sentinel: the runtime is waiting for
        /// provider connectivity to return and will retry until it
        /// does (or the run is cancelled). Render it as an indefinite
        /// "waiting for connection…" state, never as a literal
        /// `X/4294967295` counter.
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
        /// The `attempt` of the last emitted `AutoRetryStart`, so the
        /// envelope pairs even when later failures were folded into an
        /// already-open connectivity streak.
        attempt: u32,
        /// Whether a follow-up attempt eventually succeeded.
        success: bool,
        /// Last error when the retry budget ran out.
        final_error: Option<String>,
    },

    /// A provider request succeeded after moving from one OAuth account to
    /// another. Emitted only after the replacement account has produced a
    /// successful response, never while merely considering a candidate.
    AccountRotation {
        /// Stable provider identifier such as `anthropic`.
        provider: String,
        /// Bounded opaque local identifier or safe display label.
        from_account: String,
        /// Bounded opaque local identifier or safe display label.
        to_account: String,
        /// Account-local failure which initiated the handoff.
        reason: AccountRotationReason,
        /// Provider Retry-After, rounded up and capped by the host.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_seconds: Option<u32>,
    },

    /// Every OAuth account for one provider is temporarily unavailable.
    ///
    /// The retry delay is the longest remaining per-account Retry-After so a
    /// consumer can preserve retryable semantics and render an honest
    /// countdown without parsing provider log text.
    AccountPoolExhausted {
        /// Stable provider identifier such as `anthropic`.
        provider: String,
        /// Number of accounts considered unavailable in the pool.
        account_count: u32,
        /// Account-local failure which exhausted the pool.
        reason: AccountRotationReason,
        /// Longest remaining Retry-After, rounded up and capped by the host.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_seconds: Option<u32>,
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
    Cancelled {
        turn: usize,
        usage: TokenUsage,
        /// Typed explanation for why the durable task was cancelled.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<TerminalReason>,
        /// Durable task that committed this event — the cancelled
        /// root, not the promoted successor. See
        /// [`AgentEvent::with_emitter_task_id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },

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
        /// `None` on events journaled before the trigger was recorded.
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "known_or_none"
        )]
        trigger: Option<CompactionTrigger>,
        /// Engine the host configured; `None` for custom compactors and
        /// events journaled before the engine was recorded.
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "known_or_none"
        )]
        engine: Option<CompactionEngine>,
        /// Occupancy that crossed the threshold. `None` when no occupancy
        /// was compared (an overflow or a user request) and on events
        /// journaled before it was recorded. Unlike `original_tokens`, it
        /// counts the system prompt and tools as well as the history.
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "known_or_none"
        )]
        occupancy: Option<CompactionOccupancy>,
        /// Threshold the trigger compared against. `None` for custom
        /// compactors and events journaled before it was recorded.
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "known_or_none"
        )]
        threshold: Option<CompactionThreshold>,
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
        /// The child's committed turns, plus one while it runs.
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
        /// The child's finished tool calls.
        tool_count: u32,
        /// Current total tokens used by this subagent (input + output).
        total_tokens: u64,
        /// Cumulative input tokens used by the child thread.
        #[serde(default)]
        input_tokens: u64,
        /// Cumulative output tokens used by the child thread.
        #[serde(default)]
        output_tokens: u64,
        /// Cumulative cached-input tokens read by the child thread.
        #[serde(default)]
        cache_read_input_tokens: u64,
        /// Cumulative cache-creation input tokens used by the child thread.
        #[serde(default)]
        cache_creation_input_tokens: u64,
        /// The child's live state, on the periodic frames written while it
        /// runs. Boxed to keep every other event as small as before.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        liveness: Option<Box<SubagentLiveness>>,
    },

    /// An LLM call of a durable root turn closed. Committed in the
    /// transaction that closes the call's attempt row, under the turn's
    /// lease, so the events of a task are exactly its closed attempts.
    UsageProgress {
        /// The root turn task that made the call.
        task_id: String,
        /// The closed attempt's row id.
        attempt_id: String,
        /// The attempt's 1-indexed number within its task.
        attempt_number: u32,
        /// The provider identity the attempt was opened against (the API
        /// shape), not necessarily the one that served it.
        provider: String,
        /// The model the attempt requested.
        model: String,
        /// The model the provider said it served.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_model: Option<String>,
        /// The route that served the attempt, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        route_provider: Option<String>,
        /// The attempt's billed usage, failed and cancelled calls included.
        usage: TokenUsage,
        /// The billed usage of every closed attempt of `task_id`, this one
        /// included (retries and compaction calls count). The turn's
        /// `TurnComplete` usage folds the same calls in, so on a completed
        /// turn the last event's figure matches it.
        turn_usage: TokenUsage,
        /// Prompt tokens of the call's last physical request (its context
        /// occupancy), when the provider reported them. `usage.input_tokens`
        /// sums failover requests and is not an occupancy.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_tokens: Option<u32>,
    },
}

impl AgentEvent {
    #[must_use]
    pub const fn thread_created(
        thread_id: ThreadId,
        source_thread_id: Option<ThreadId>,
        fork_after_committed_turns: Option<u32>,
    ) -> Self {
        Self::ThreadCreated {
            thread_id,
            source_thread_id,
            fork_after_committed_turns,
        }
    }

    #[must_use]
    pub const fn start(thread_id: ThreadId, turn: usize) -> Self {
        Self::Start {
            thread_id,
            turn,
            emitter_task_id: None,
        }
    }

    /// Attribute an event to the durable task whose execution
    /// committed it (the task id in its string form).
    ///
    /// The lifecycle variants (`Start`, `TurnComplete`, `Done`,
    /// `BudgetExceeded`, `Error`, `Cancelled`) and the content variants
    /// (`Thinking`, `ThinkingDelta`, `TextDelta`, `Text`) carry the
    /// attribution; every other variant is returned unchanged.
    /// Attribution is always the emitter's own identity, never a
    /// successor's: a cancelled root's late salvage commit — content
    /// included — still names the cancelled root, which is what lets a
    /// reader tell a superseded frame apart from the thread's live one
    /// (and lets the ACP run loop drop a predecessor's stale deltas
    /// instead of rendering them inside the successor's answer).
    ///
    /// Runs without a durable task behind them (the embedded SDK loop)
    /// leave the field `None`, as do events journaled before the field
    /// existed.
    #[must_use]
    pub fn with_emitter_task_id(mut self, task_id: impl Into<String>) -> Self {
        let task_id = task_id.into();
        match &mut self {
            Self::Start {
                emitter_task_id, ..
            }
            | Self::UserInput {
                emitter_task_id, ..
            }
            | Self::TurnComplete {
                emitter_task_id, ..
            }
            | Self::Done {
                emitter_task_id, ..
            }
            | Self::BudgetExceeded {
                emitter_task_id, ..
            }
            | Self::Error {
                emitter_task_id, ..
            }
            | Self::Cancelled {
                emitter_task_id, ..
            }
            | Self::Thinking {
                emitter_task_id, ..
            }
            | Self::ThinkingDelta {
                emitter_task_id, ..
            }
            | Self::TextDelta {
                emitter_task_id, ..
            }
            | Self::Text {
                emitter_task_id, ..
            } => *emitter_task_id = Some(task_id),
            _ => {}
        }
        self
    }

    /// The durable task that committed this event, when the event is a
    /// lifecycle or content variant that was stamped. See
    /// [`AgentEvent::with_emitter_task_id`].
    #[must_use]
    pub fn emitter_task_id(&self) -> Option<&str> {
        match self {
            Self::Start {
                emitter_task_id, ..
            }
            | Self::UserInput {
                emitter_task_id, ..
            }
            | Self::TurnComplete {
                emitter_task_id, ..
            }
            | Self::Done {
                emitter_task_id, ..
            }
            | Self::BudgetExceeded {
                emitter_task_id, ..
            }
            | Self::Error {
                emitter_task_id, ..
            }
            | Self::Cancelled {
                emitter_task_id, ..
            }
            | Self::Thinking {
                emitter_task_id, ..
            }
            | Self::ThinkingDelta {
                emitter_task_id, ..
            }
            | Self::TextDelta {
                emitter_task_id, ..
            }
            | Self::Text {
                emitter_task_id, ..
            } => emitter_task_id.as_deref(),
            _ => None,
        }
    }

    #[must_use]
    pub const fn user_input(thread_id: ThreadId, content: Vec<ContentBlock>) -> Self {
        Self::UserInput {
            thread_id,
            content,
            emitter_task_id: None,
        }
    }

    #[must_use]
    pub fn thinking(message_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Thinking {
            message_id: message_id.into(),
            text: text.into(),
            emitter_task_id: None,
        }
    }

    #[must_use]
    pub fn thinking_delta(message_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self::ThinkingDelta {
            message_id: message_id.into(),
            delta: delta.into(),
            emitter_task_id: None,
        }
    }

    #[must_use]
    pub fn text_delta(message_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self::TextDelta {
            message_id: message_id.into(),
            delta: delta.into(),
            emitter_task_id: None,
        }
    }

    #[must_use]
    pub fn text(message_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Text {
            message_id: message_id.into(),
            text: text.into(),
            emitter_task_id: None,
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

    /// Build the durable event that asks a client to render the parked
    /// question batch.
    #[must_use]
    pub fn question_asked(
        task_id: impl Into<String>,
        questions: Vec<crate::QuestionPayload>,
    ) -> Self {
        Self::QuestionAsked {
            task_id: task_id.into(),
            questions,
        }
    }

    #[must_use]
    pub const fn turn_complete(turn: usize, usage: TokenUsage) -> Self {
        Self::TurnComplete {
            turn,
            usage,
            emitter_task_id: None,
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
            estimated_cost_usd: None,
            emitter_task_id: None,
        }
    }

    #[must_use]
    pub const fn done_with_cost(
        thread_id: ThreadId,
        total_turns: usize,
        total_usage: TokenUsage,
        duration: Duration,
        estimated_cost_usd: Option<f64>,
    ) -> Self {
        Self::Done {
            thread_id,
            total_turns,
            total_usage,
            duration,
            estimated_cost_usd,
            emitter_task_id: None,
        }
    }

    #[must_use]
    pub const fn budget_exceeded(
        thread_id: ThreadId,
        total_turns: usize,
        total_usage: TokenUsage,
        duration: Duration,
        estimated_cost_usd: Option<f64>,
        limit: BudgetLimitKind,
    ) -> Self {
        Self::BudgetExceeded {
            thread_id,
            total_turns,
            total_usage,
            duration,
            estimated_cost_usd,
            limit,
            emitter_task_id: None,
        }
    }

    #[must_use]
    pub fn error(message: impl Into<String>, recoverable: bool) -> Self {
        Self::Error {
            message: message.into(),
            recoverable,
            reason: None,
            emitter_task_id: None,
        }
    }

    /// Build a terminal error event with its durable task reason attached.
    #[must_use]
    pub fn terminal_error(message: impl Into<String>, reason: TerminalReason) -> Self {
        Self::Error {
            message: message.into(),
            recoverable: false,
            reason: Some(reason),
            emitter_task_id: None,
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
        Self::cancelled_with_reason(turn, usage, TerminalReason::UserCancel)
    }

    /// Build a terminal cancellation event with its durable task reason.
    #[must_use]
    pub const fn cancelled_with_reason(
        turn: usize,
        usage: TokenUsage,
        reason: TerminalReason,
    ) -> Self {
        Self::Cancelled {
            turn,
            usage,
            reason: Some(reason),
            emitter_task_id: None,
        }
    }

    #[must_use]
    pub const fn context_compacted(
        original_count: usize,
        new_count: usize,
        original_tokens: usize,
        new_tokens: usize,
        trigger: CompactionTrigger,
        engine: Option<CompactionEngine>,
    ) -> Self {
        Self::ContextCompacted {
            original_count,
            new_count,
            original_tokens,
            new_tokens,
            trigger: Some(trigger),
            engine,
            occupancy: None,
            threshold: None,
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
            AgentEvent::Text {
                message_id, text, ..
            } => {
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
            AgentEvent::Text {
                message_id, text, ..
            } => {
                assert_eq!(message_id, "msg_1");
                assert_eq!(text, "hello");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn account_pool_events_roundtrip_their_stable_wire_shape() {
        let rotation = AgentEvent::AccountRotation {
            provider: "anthropic".to_owned(),
            from_account: "account-a".to_owned(),
            to_account: "account-b".to_owned(),
            reason: AccountRotationReason::RateLimited,
            retry_after_seconds: Some(42),
        };
        let json = serde_json::to_value(&rotation).expect("serialize rotation");
        assert_eq!(
            json,
            serde_json::json!({
                "type": "account_rotation",
                "provider": "anthropic",
                "from_account": "account-a",
                "to_account": "account-b",
                "reason": "rate_limited",
                "retry_after_seconds": 42,
            })
        );
        let restored: AgentEvent = serde_json::from_value(json).expect("deserialize rotation");
        assert!(matches!(
            restored,
            AgentEvent::AccountRotation {
                reason: AccountRotationReason::RateLimited,
                retry_after_seconds: Some(42),
                ..
            }
        ));

        let exhausted = AgentEvent::AccountPoolExhausted {
            provider: "anthropic".to_owned(),
            account_count: 3,
            reason: AccountRotationReason::RateLimited,
            retry_after_seconds: Some(90),
        };
        let json = serde_json::to_value(&exhausted).expect("serialize exhaustion");
        assert_eq!(
            json,
            serde_json::json!({
                "type": "account_pool_exhausted",
                "provider": "anthropic",
                "account_count": 3,
                "reason": "rate_limited",
                "retry_after_seconds": 90,
            })
        );
        let restored: AgentEvent = serde_json::from_value(json).expect("deserialize exhaustion");
        assert!(matches!(
            restored,
            AgentEvent::AccountPoolExhausted {
                account_count: 3,
                retry_after_seconds: Some(90),
                ..
            }
        ));
    }

    #[test]
    fn account_rotation_reason_has_a_forward_compatible_unknown_sink() {
        let event: AgentEvent = serde_json::from_value(serde_json::json!({
            "type": "account_rotation",
            "provider": "anthropic",
            "from_account": "account-a",
            "to_account": "account-b",
            "reason": "provider_policy",
        }))
        .expect("unknown reason remains replayable");

        assert!(matches!(
            event,
            AgentEvent::AccountRotation {
                reason: AccountRotationReason::Unknown,
                retry_after_seconds: None,
                ..
            }
        ));
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

    #[test]
    fn done_event_serializes_duration_as_millis() -> serde_json::Result<()> {
        let seq = SequenceCounter::new();
        let envelope = AgentEventEnvelope::wrap(
            AgentEvent::done(
                ThreadId::from_string("t"),
                3,
                TokenUsage::default(),
                Duration::from_millis(2500),
            ),
            &seq,
        );
        let json = serde_json::to_value(&envelope)?;

        // Flat millisecond integer, matching `TurnSummary::duration_ms` — not
        // the old nested `{"secs":..,"nanos":..}` object under `duration`.
        assert_eq!(
            json.get("duration_ms").and_then(serde_json::Value::as_u64),
            Some(2500)
        );
        assert!(
            json.get("duration").is_none(),
            "old `duration` key must be gone: {json}"
        );

        let restored: AgentEventEnvelope = serde_json::from_value(json)?;
        match restored.event {
            AgentEvent::Done { duration, .. } => {
                assert_eq!(duration, Duration::from_millis(2500));
            }
            other => panic!("expected Done, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn done_event_deserializes_legacy_duration_object() -> serde_json::Result<()> {
        // Durable rows written before the wire form changed to `duration_ms`
        // carry serde's default object under the old `duration` key. Hosts
        // replay them with `serde_json::from_value`; they must keep
        // decoding after an upgrade.
        let legacy = serde_json::json!({
            "type": "done",
            "thread_id": "t-legacy",
            "total_turns": 3,
            "total_usage": TokenUsage::default(),
            "duration": { "secs": 2, "nanos": 500_000_000 },
        });
        let event: AgentEvent = serde_json::from_value(legacy)?;
        match event {
            AgentEvent::Done {
                duration,
                total_turns,
                ..
            } => {
                assert_eq!(duration, Duration::from_millis(2500));
                assert_eq!(total_turns, 3);
            }
            other => panic!("expected Done, got {other:?}"),
        }

        // The current flat-millis form decodes too...
        let current = serde_json::json!({
            "type": "done",
            "thread_id": "t-current",
            "total_turns": 3,
            "total_usage": TokenUsage::default(),
            "duration_ms": 2500,
        });
        let event: AgentEvent = serde_json::from_value(current)?;
        let AgentEvent::Done { duration, .. } = event else {
            panic!("expected Done");
        };
        assert_eq!(duration, Duration::from_millis(2500));

        // ...and a re-serialized legacy event is normalized to millis.
        let legacy_event: AgentEvent = serde_json::from_value(serde_json::json!({
            "type": "done",
            "thread_id": "t-roundtrip",
            "total_turns": 1,
            "total_usage": TokenUsage::default(),
            "duration": { "secs": 1, "nanos": 0 },
        }))?;
        let reserialized = serde_json::to_value(&legacy_event)?;
        assert_eq!(
            reserialized
                .get("duration_ms")
                .and_then(serde_json::Value::as_u64),
            Some(1000),
            "round-trips must write the millis form: {reserialized}"
        );
        assert!(reserialized.get("duration").is_none());
        Ok(())
    }

    #[test]
    fn context_compacted_rows_without_trigger_or_engine_still_deserialize() -> serde_json::Result<()>
    {
        let legacy: AgentEvent = serde_json::from_value(serde_json::json!({
            "type": "context_compacted",
            "original_count": 10,
            "new_count": 4,
            "original_tokens": 1_000,
            "new_tokens": 400,
        }))?;
        assert!(
            matches!(
                legacy,
                AgentEvent::ContextCompacted {
                    original_count: 10,
                    new_tokens: 400,
                    trigger: None,
                    engine: None,
                    ..
                }
            ),
            "{legacy:?}"
        );
        Ok(())
    }

    #[test]
    fn context_compacted_records_trigger_and_engine_and_tolerates_unknown_values()
    -> serde_json::Result<()> {
        let event = AgentEvent::context_compacted(
            10,
            4,
            1_000,
            400,
            CompactionTrigger::UserRequested,
            Some(CompactionEngine::Snapcompact),
        );
        let encoded = serde_json::to_value(&event)?;
        assert_eq!(encoded["trigger"], "user_requested");
        assert_eq!(encoded["engine"], "snapcompact");
        let restored: AgentEvent = serde_json::from_value(encoded)?;
        assert!(
            matches!(
                restored,
                AgentEvent::ContextCompacted {
                    trigger: Some(CompactionTrigger::UserRequested),
                    engine: Some(CompactionEngine::Snapcompact),
                    ..
                }
            ),
            "{restored:?}"
        );

        let future: AgentEvent = serde_json::from_value(serde_json::json!({
            "type": "context_compacted",
            "original_count": 10,
            "new_count": 4,
            "original_tokens": 1_000,
            "new_tokens": 400,
            "trigger": "scheduled",
            "engine": "streaming_summary",
        }))?;
        assert!(
            matches!(
                future,
                AgentEvent::ContextCompacted {
                    trigger: None,
                    engine: None,
                    ..
                }
            ),
            "{future:?}"
        );
        Ok(())
    }

    #[test]
    fn context_compacted_records_occupancy_and_threshold() -> serde_json::Result<()> {
        let event = AgentEvent::ContextCompacted {
            original_count: 21,
            new_count: 21,
            original_tokens: 28_379,
            new_tokens: 15_000,
            trigger: Some(CompactionTrigger::Threshold),
            engine: Some(CompactionEngine::PruneFirst),
            occupancy: Some(CompactionOccupancy {
                tokens: 83_414,
                source: OccupancySource::Measured,
            }),
            threshold: Some(CompactionThreshold {
                tokens: 136_000,
                source: ThresholdSource::ModelAware,
                context_window_tokens: Some(200_000),
            }),
        };
        let encoded = serde_json::to_value(&event)?;
        assert_eq!(
            encoded["occupancy"],
            serde_json::json!({"tokens": 83_414, "source": "measured"})
        );
        assert_eq!(
            encoded["threshold"],
            serde_json::json!({
                "tokens": 136_000,
                "source": "model_aware",
                "context_window_tokens": 200_000,
            })
        );
        let restored: AgentEvent = serde_json::from_value(encoded)?;
        assert_eq!(
            serde_json::to_value(&restored)?,
            serde_json::to_value(&event)?
        );

        let future: AgentEvent = serde_json::from_value(serde_json::json!({
            "type": "context_compacted",
            "original_count": 21,
            "new_count": 12,
            "original_tokens": 28_379,
            "new_tokens": 1_514,
            "occupancy": {"tokens": 83_414, "source": "provider_counted", "margin": 3},
            "threshold": {"tokens": 80_000, "source": "operator_pinned"},
        }))?;
        assert!(
            matches!(
                future,
                AgentEvent::ContextCompacted {
                    occupancy: Some(CompactionOccupancy {
                        tokens: 83_414,
                        source: OccupancySource::Unknown,
                    }),
                    threshold: Some(CompactionThreshold {
                        tokens: 80_000,
                        source: ThresholdSource::Unknown,
                        context_window_tokens: None,
                    }),
                    ..
                }
            ),
            "{future:?}"
        );

        let legacy: AgentEvent = serde_json::from_value(serde_json::json!({
            "type": "context_compacted",
            "original_count": 21,
            "new_count": 12,
            "original_tokens": 28_379,
            "new_tokens": 1_514,
            "trigger": "threshold",
            "engine": "prune_first",
        }))?;
        assert!(
            matches!(
                legacy,
                AgentEvent::ContextCompacted {
                    occupancy: None,
                    threshold: None,
                    ..
                }
            ),
            "{legacy:?}"
        );
        let reencoded = serde_json::to_value(&legacy)?;
        assert!(reencoded.get("occupancy").is_none());
        assert!(reencoded.get("threshold").is_none());
        Ok(())
    }

    #[test]
    fn terminal_reason_round_trips_provider_kind_and_accepts_future_variants()
    -> serde_json::Result<()> {
        let provider = TerminalReason::ProviderError {
            kind: "rate_limited".to_owned(),
        };
        let encoded = serde_json::to_value(&provider)?;
        assert_eq!(
            encoded,
            serde_json::json!({
                "reason": "provider_error",
                "kind": "rate_limited",
            }),
        );
        let restored: TerminalReason = serde_json::from_value(encoded)?;
        assert_eq!(restored, provider);

        let future: TerminalReason = serde_json::from_value(serde_json::json!({
            "reason": "provider_shutdown",
            "retryable": false,
        }))?;
        assert_eq!(future, TerminalReason::Unknown);
        Ok(())
    }

    #[test]
    fn subagent_progress_deserializes_legacy_rows_without_usage_breakdown() -> serde_json::Result<()>
    {
        let legacy = serde_json::json!({
            "type": "subagent_progress",
            "subagent_id": "call-1",
            "subagent_name": "explore",
            "nickname": null,
            "child_thread_id": null,
            "child_root_task_id": null,
            "subagent_task_id": null,
            "max_turns": 3,
            "current_turn": 1,
            "model": "mock",
            "tool_name": "explore",
            "tool_context": "inspect",
            "completed": false,
            "success": false,
            "tool_count": 0,
            "total_tokens": 12,
        });
        let event: AgentEvent = serde_json::from_value(legacy)?;
        match event {
            AgentEvent::SubagentProgress {
                total_tokens,
                input_tokens,
                output_tokens,
                cache_read_input_tokens,
                cache_creation_input_tokens,
                ..
            } => {
                assert_eq!(total_tokens, 12);
                assert_eq!(input_tokens, 0);
                assert_eq!(output_tokens, 0);
                assert_eq!(cache_read_input_tokens, 0);
                assert_eq!(cache_creation_input_tokens, 0);
            }
            other => panic!("expected SubagentProgress, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn usage_progress_round_trips_and_its_optional_fields_default() -> serde_json::Result<()> {
        let row = serde_json::json!({
            "type": "usage_progress",
            "task_id": "task_1",
            "attempt_id": "attempt_1",
            "attempt_number": 2,
            "provider": "anthropic",
            "model": "claude-sonnet-5-5",
            "usage": {"input_tokens": 120, "output_tokens": 7},
            "turn_usage": {
                "input_tokens": 300,
                "output_tokens": 20,
                "cached_input_tokens": 100,
                "cache_creation_input_tokens": 50
            },
        });
        let bare: AgentEvent = serde_json::from_value(row)?;
        let AgentEvent::UsageProgress {
            attempt_number,
            response_model: None,
            route_provider: None,
            context_tokens: None,
            usage,
            turn_usage,
            ..
        } = &bare
        else {
            panic!("expected a UsageProgress without its optional fields, got {bare:?}");
        };
        assert_eq!(*attempt_number, 2);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(turn_usage.cache_creation_input_tokens, 50);
        let rewritten = serde_json::to_value(&bare)?;
        for absent in ["response_model", "route_provider", "context_tokens"] {
            assert!(
                rewritten.get(absent).is_none(),
                "{absent} written: {rewritten}"
            );
        }

        let full = AgentEvent::UsageProgress {
            task_id: "task_1".into(),
            attempt_id: "attempt_1".into(),
            attempt_number: 2,
            provider: "anthropic".into(),
            model: "claude-sonnet-5-5".into(),
            response_model: Some("claude-sonnet-5-5-20261001".into()),
            route_provider: Some("openrouter".into()),
            usage: TokenUsage {
                input_tokens: 120,
                output_tokens: 7,
                cached_input_tokens: 30,
                cache_creation_input_tokens: 11,
            },
            turn_usage: TokenUsage::default(),
            context_tokens: Some(120),
        };
        let json = serde_json::to_value(&full)?;
        assert_eq!(json["type"], "usage_progress");
        assert_eq!(
            serde_json::to_value(serde_json::from_value::<AgentEvent>(json.clone())?)?,
            json
        );
        Ok(())
    }

    #[test]
    fn subagent_liveness_is_additive_on_the_journal_row() -> serde_json::Result<()> {
        let mut row = serde_json::json!({
            "type": "subagent_progress",
            "subagent_id": "call-1",
            "subagent_name": "explore",
            "nickname": null,
            "child_thread_id": null,
            "child_root_task_id": null,
            "subagent_task_id": null,
            "max_turns": 3,
            "current_turn": 1,
            "model": "mock",
            "tool_name": "explore",
            "tool_context": "inspect",
            "completed": false,
            "success": false,
            "tool_count": 0,
            "total_tokens": 12,
        });
        let legacy: AgentEvent = serde_json::from_value(row.clone())?;
        assert!(matches!(
            legacy,
            AgentEvent::SubagentProgress { liveness: None, .. }
        ));
        let rewritten = serde_json::to_value(&legacy)?;
        assert!(
            rewritten.get("liveness").is_none(),
            "a frame without liveness writes no key: {rewritten}"
        );

        row["liveness"] = serde_json::json!({
            "last_activity_at": "2026-10-18T12:00:00Z",
            "phase": "waiting_on_tools",
            "llm_calls": 2,
            "output_tokens_live": 40,
        });
        let live: AgentEvent = serde_json::from_value(row.clone())?;
        let AgentEvent::SubagentProgress {
            liveness: Some(liveness),
            ..
        } = &live
        else {
            panic!("expected a live SubagentProgress, got {live:?}");
        };
        assert_eq!(liveness.phase, SubagentPhase::WaitingOnTools);
        assert_eq!(liveness.llm_calls, 2);
        assert_eq!(liveness.output_tokens_live, 40);
        assert_eq!(
            liveness
                .last_activity_at
                .map(OffsetDateTime::unix_timestamp),
            Some(1_792_324_800)
        );
        assert_eq!(serde_json::to_value(&live)?["liveness"], row["liveness"]);

        row["liveness"]["phase"] = serde_json::json!("finalizing");
        let newer: AgentEvent = serde_json::from_value(row)?;
        assert!(matches!(
            newer,
            AgentEvent::SubagentProgress {
                liveness: Some(ref liveness),
                ..
            } if liveness.phase == SubagentPhase::Unknown
        ));
        Ok(())
    }

    #[test]
    fn budget_exceeded_event_deserializes_legacy_duration_object() -> serde_json::Result<()> {
        // BudgetExceeded has no pre-rename durable rows, but the field uses
        // the same adapter — keep it uniformly lenient.
        let legacy = serde_json::json!({
            "type": "budget_exceeded",
            "thread_id": "t-legacy",
            "total_turns": 2,
            "total_usage": TokenUsage::default(),
            "duration": { "secs": 1, "nanos": 250_000_000 },
            "limit": "total_tokens",
        });
        let event: AgentEvent = serde_json::from_value(legacy)?;
        let AgentEvent::BudgetExceeded { duration, .. } = event else {
            panic!("expected BudgetExceeded");
        };
        assert_eq!(duration, Duration::from_millis(1250));
        Ok(())
    }

    #[test]
    fn budget_exceeded_event_serializes_duration_as_millis() -> serde_json::Result<()> {
        let seq = SequenceCounter::new();
        let envelope = AgentEventEnvelope::wrap(
            AgentEvent::budget_exceeded(
                ThreadId::from_string("t"),
                2,
                TokenUsage::default(),
                Duration::from_millis(1200),
                Some(0.5),
                BudgetLimitKind::TotalTokens,
            ),
            &seq,
        );
        let json = serde_json::to_value(&envelope)?;

        // Flat millisecond integer, mirroring the `Done` wire form.
        assert_eq!(
            json.get("duration_ms").and_then(serde_json::Value::as_u64),
            Some(1200)
        );
        assert!(
            json.get("duration").is_none(),
            "no nested `duration` key expected: {json}"
        );

        let restored: AgentEventEnvelope = serde_json::from_value(json)?;
        match restored.event {
            AgentEvent::BudgetExceeded { duration, .. } => {
                assert_eq!(duration, Duration::from_millis(1200));
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
        Ok(())
    }

    /// One representative value of every [`AgentEvent`] variant, so the
    /// envelope round-trip test exercises the full streaming contract.
    ///
    /// The variants are produced by a few cohesive builders concatenated
    /// in the original order, so the round-trip test still sees the exact
    /// same set of values.
    fn sample_all_variants() -> Vec<AgentEvent> {
        let thread = ThreadId::from_string("thread-1");
        let usage = TokenUsage::default();
        let mut events = session_open_events(&thread);
        events.extend(streamed_content_events());
        events.extend(tool_call_events());
        events.extend(turn_completion_events(&thread, &usage));
        events.extend(failure_and_retry_events());
        events.extend(auxiliary_events(&usage));
        events
    }

    /// `Start` / `UserInput`: the events opening a thread turn.
    fn session_open_events(thread: &ThreadId) -> Vec<AgentEvent> {
        vec![
            AgentEvent::ThreadCreated {
                thread_id: thread.clone(),
                source_thread_id: None,
                fork_after_committed_turns: None,
            },
            AgentEvent::Start {
                thread_id: thread.clone(),
                turn: 1,
                emitter_task_id: Some("task-start".into()),
            },
            AgentEvent::UserInput {
                thread_id: thread.clone(),
                content: vec![ContentBlock::Text { text: "hi".into() }],
                emitter_task_id: None,
            },
        ]
    }

    /// Streamed assistant content: consolidated and delta forms of
    /// thinking and text.
    fn streamed_content_events() -> Vec<AgentEvent> {
        vec![
            AgentEvent::Thinking {
                message_id: "m".into(),
                text: "t".into(),
                emitter_task_id: None,
            },
            AgentEvent::ThinkingDelta {
                message_id: "m".into(),
                delta: "d".into(),
                emitter_task_id: None,
            },
            AgentEvent::TextDelta {
                message_id: "m".into(),
                delta: "d".into(),
                emitter_task_id: None,
            },
            AgentEvent::Text {
                message_id: "m".into(),
                text: "t".into(),
                emitter_task_id: None,
            },
        ]
    }

    /// Tool-call lifecycle: start, end, progress, and confirmation.
    fn tool_call_events() -> Vec<AgentEvent> {
        vec![
            AgentEvent::ToolCallStart {
                id: "id".into(),
                name: "n".into(),
                display_name: "N".into(),
                input: serde_json::json!({}),
                tier: ToolTier::Observe,
            },
            AgentEvent::ToolCallEnd {
                id: "id".into(),
                name: "n".into(),
                display_name: "N".into(),
                result: ToolResult::success("ok"),
            },
            AgentEvent::ToolProgress {
                id: "id".into(),
                name: "n".into(),
                display_name: "N".into(),
                stage: "s".into(),
                message: "m".into(),
                data: None,
            },
            AgentEvent::ToolRequiresConfirmation {
                id: "id".into(),
                name: "n".into(),
                display_name: "N".into(),
                input: serde_json::json!({}),
                description: "d".into(),
            },
        ]
    }

    /// Turn-completion summaries: `TurnComplete` and the terminal `Done`.
    fn turn_completion_events(thread: &ThreadId, usage: &TokenUsage) -> Vec<AgentEvent> {
        vec![
            AgentEvent::TurnComplete {
                turn: 1,
                usage: usage.clone(),
                emitter_task_id: Some("task-turn-complete".into()),
            },
            AgentEvent::Done {
                thread_id: thread.clone(),
                total_turns: 2,
                total_usage: usage.clone(),
                duration: Duration::from_millis(1500),
                estimated_cost_usd: Some(0.0123),
                emitter_task_id: Some("task-done".into()),
            },
        ]
    }

    /// Error and auto-retry signalling events.
    fn failure_and_retry_events() -> Vec<AgentEvent> {
        vec![
            AgentEvent::Error {
                message: "e".into(),
                recoverable: true,
                reason: None,
                emitter_task_id: Some("task-error".into()),
            },
            AgentEvent::AutoRetryStart {
                attempt: 1,
                max_attempts: 5,
                delay_ms: 100,
                error_message: "rate limited".into(),
            },
            AgentEvent::AutoRetryEnd {
                attempt: 1,
                success: true,
                final_error: None,
            },
        ]
    }

    /// Remaining auxiliary events: refusal, cancellation, compaction,
    /// and subagent progress.
    fn auxiliary_events(usage: &TokenUsage) -> Vec<AgentEvent> {
        vec![
            AgentEvent::Refusal {
                message_id: "m".into(),
                text: Some("no".into()),
            },
            AgentEvent::Cancelled {
                turn: 1,
                usage: usage.clone(),
                reason: Some(TerminalReason::UserCancel),
                emitter_task_id: Some("task-cancelled".into()),
            },
            AgentEvent::BudgetExceeded {
                thread_id: ThreadId::from_string("thread-1"),
                total_turns: 3,
                total_usage: usage.clone(),
                duration: Duration::from_millis(750),
                estimated_cost_usd: Some(0.5),
                limit: BudgetLimitKind::CostUsd,
                emitter_task_id: Some("task-budget".into()),
            },
            AgentEvent::ContextCompacted {
                original_count: 10,
                new_count: 5,
                original_tokens: 100,
                new_tokens: 50,
                trigger: Some(CompactionTrigger::Overflow),
                engine: Some(CompactionEngine::PruneFirst),
                occupancy: Some(CompactionOccupancy {
                    tokens: 83_414,
                    source: OccupancySource::Measured,
                }),
                threshold: Some(CompactionThreshold {
                    tokens: 80_000,
                    source: ThresholdSource::FixedUnknownModel,
                    context_window_tokens: None,
                }),
            },
            AgentEvent::SubagentProgress {
                subagent_id: "s".into(),
                subagent_name: "explore".into(),
                nickname: None,
                child_thread_id: None,
                child_root_task_id: None,
                subagent_task_id: None,
                max_turns: None,
                current_turn: None,
                model: None,
                tool_name: "t".into(),
                tool_context: "c".into(),
                completed: false,
                success: false,
                tool_count: 0,
                total_tokens: 0,
                input_tokens: 0,
                output_tokens: 0,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                liveness: None,
            },
        ]
    }

    // ===================
    // Emitter task identity
    // ===================

    #[test]
    fn emitter_task_id_is_absent_from_journal_rows_written_before_the_field()
    -> serde_json::Result<()> {
        // Durable rows predating the field carry no `emitter_task_id`
        // key; they must decode (as `None`), not fail the whole thread's
        // replay.
        let legacy = serde_json::json!({
            "type": "done",
            "thread_id": "t-legacy",
            "total_turns": 2,
            "total_usage": TokenUsage::default(),
            "duration_ms": 1000,
        });
        let event: AgentEvent = serde_json::from_value(legacy)?;
        assert_eq!(event.emitter_task_id(), None);

        // And an unstamped event never writes the key, so the wire form
        // stays byte-identical for consumers that predate it.
        let json = serde_json::to_value(&event)?;
        assert!(
            json.get("emitter_task_id").is_none(),
            "unstamped events must omit the key: {json}"
        );
        Ok(())
    }

    #[test]
    fn with_emitter_task_id_stamps_every_lifecycle_variant() -> serde_json::Result<()> {
        let thread = ThreadId::from_string("t");
        let usage = TokenUsage::default();
        let lifecycle = vec![
            AgentEvent::start(thread.clone(), 1),
            AgentEvent::TurnComplete {
                turn: 1,
                usage: usage.clone(),
                emitter_task_id: None,
            },
            AgentEvent::done(thread.clone(), 1, usage.clone(), Duration::from_secs(1)),
            AgentEvent::budget_exceeded(
                thread,
                1,
                usage.clone(),
                Duration::from_secs(1),
                None,
                BudgetLimitKind::TotalTokens,
            ),
            AgentEvent::error("boom", false),
            AgentEvent::cancelled(1, usage),
        ];
        for event in lifecycle {
            let label = format!("{event:?}");
            assert_eq!(event.emitter_task_id(), None, "{label}: starts unstamped");

            let stamped = event.with_emitter_task_id("task-42");
            assert_eq!(stamped.emitter_task_id(), Some("task-42"), "{label}");

            let json = serde_json::to_value(&stamped)?;
            assert_eq!(
                json.get("emitter_task_id")
                    .and_then(serde_json::Value::as_str),
                Some("task-42"),
                "{label}: stamped events carry the key: {json}"
            );
            let restored: AgentEvent = serde_json::from_value(json)?;
            assert_eq!(restored.emitter_task_id(), Some("task-42"), "{label}");
        }
        Ok(())
    }

    #[test]
    fn with_emitter_task_id_covers_content_but_not_tool_frames() -> serde_json::Result<()> {
        // Attribution covers lifecycle AND content variants (ENG-9422:
        // adjacency pairing broke exactly when a cancelled attempt's
        // salvage flush landed deltas after the successor's Start).
        let text = AgentEvent::text("m", "hi").with_emitter_task_id("task-42");
        assert_eq!(text.emitter_task_id(), Some("task-42"));

        // Unattributed content stays byte-identical on the wire.
        let bare = AgentEvent::text("m", "hi");
        let json = serde_json::to_value(&bare)?;
        assert!(
            json.get("emitter_task_id").is_none(),
            "unattributed content must not grow the key: {json}"
        );

        // Tool frames still pair by their own call id, not attribution.
        let tool = AgentEvent::tool_call_start(
            "t1",
            "grep",
            "Grep",
            serde_json::json!({}),
            ToolTier::Observe,
        )
        .with_emitter_task_id("task-42");
        assert_eq!(tool.emitter_task_id(), None);
        Ok(())
    }

    #[test]
    fn every_variant_envelope_has_flat_keys_and_round_trips() -> serde_json::Result<()> {
        let seq = SequenceCounter::new();
        for event in sample_all_variants() {
            let label = format!("{event:?}");
            let envelope = AgentEventEnvelope::wrap(event, &seq);
            let json = serde_json::to_value(&envelope)?;

            // Envelope metadata + the event discriminant are all flat keys.
            for key in ["event_id", "sequence", "timestamp", "type"] {
                assert!(
                    json.get(key).is_some(),
                    "{label}: missing flat key `{key}` in {json}"
                );
            }
            // The `#[serde(flatten)]` must not leave a nested wrapper, and no
            // variant field may collide with an envelope key.
            assert!(
                json.get("event").is_none(),
                "{label}: unexpected nested `event` key in {json}"
            );

            let restored: AgentEventEnvelope = serde_json::from_value(json.clone())?;
            assert_eq!(
                serde_json::to_value(&restored)?,
                json,
                "{label}: envelope round-trip changed the wire form"
            );
        }
        Ok(())
    }
}
