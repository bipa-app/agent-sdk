//! The session view: a per-thread projection of the committed event log.
//!
//! [`fold`] is a pure function of the view's [`ThreadViewStatus`], one
//! committed event and the host's [`ViewConfig`]. It returns the next status
//! and the [`ViewOp`]s that turn the previous view into the next one. A store
//! applies the ops in the same transaction as the event; a client applies the
//! same ops to a snapshot. Neither interprets events itself.
//!
//! `fold` reads the status only, never the items: every op addresses an item
//! by a stable key, so a backend applies it as a keyed upsert.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::ThreadId;
use crate::events::{
    AccountRotationReason, AgentEvent, CompactionEngine, CompactionOccupancy, CompactionThreshold,
    CompactionTrigger, TerminalReason,
};
use crate::llm::ContentBlock;
use crate::types::{BudgetLimitKind, QuestionPayload, TokenUsage, ToolResult, ToolTier};

/// Version of the fold's semantics. A stored view at another version is
/// treated as not materialized and is rebuilt from the events.
pub const VIEW_SCHEMA: u32 = 1;

/// How many recently closed message and tool keys the status remembers, so a
/// late final event patches its item instead of opening a second one.
pub const RECENTLY_CLOSED_LIMIT: usize = 32;

/// Host settings for the view. A value, so a store can hold it and pass it to
/// [`fold`] inside its transaction.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewConfig {
    /// Whether the store maintains views. Off by default: a host with no
    /// view consumer writes no view rows.
    pub enabled: bool,
    /// A `ToolProgress` with this tool name is the host's stall notice. It
    /// sets [`ThreadViewStatus::stalled`] and adds no item.
    pub stall_tool_name: Option<String>,
    /// Document kinds the view carries in its status.
    pub visible_document_kinds: Vec<String>,
}

/// Whether the thread's root turn is running.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TurnState {
    #[default]
    Idle,
    Running {
        turn: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<String>,
    },
}

/// How the last root turn ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TurnOutcome {
    Done {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        estimated_cost_usd: Option<f64>,
    },
    Cancelled {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<TerminalReason>,
    },
    BudgetExceeded {
        limit: BudgetLimitKind,
    },
    Failed {
        message: String,
        reason: TerminalReason,
    },
}

/// An open auto-retry. `max_attempts == u32::MAX` is the indefinite wait for
/// provider connectivity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryState {
    pub attempt: u32,
    pub max_attempts: u32,
    pub delay_ms: u64,
    pub error_message: String,
    #[serde(with = "time::serde::rfc3339")]
    pub started_at: OffsetDateTime,
}

impl RetryState {
    #[must_use]
    pub const fn is_connectivity_wait(&self) -> bool {
        self.max_attempts == u32::MAX
    }
}

/// Token totals the view reports.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageView {
    /// Usage of the last completed LLM round trip.
    pub last_turn: TokenUsage,
    /// Turns completed in the current or last run.
    pub turns: usize,
    /// Run totals from the last `Done`, `BudgetExceeded` or `Cancelled`.
    pub run: TokenUsage,
}

/// A confirmation the thread waits on. `relayed` marks one copied from
/// another thread: it has no `ToolCallStart` here and is not cleared by this
/// thread's turn boundaries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingConfirmation {
    pub key: String,
    pub relayed: bool,
}

/// The small, bounded part of the view.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadViewStatus {
    pub schema: u32,
    /// Sequence of the last event folded.
    pub sequence: u64,
    pub turn: TurnState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_outcome: Option<TurnOutcome>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_messages: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recently_closed: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_confirmations: Vec<PendingConfirmation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_questions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryState>,
    pub usage: UsageView,
    #[serde(default)]
    pub stalled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forked_from: Option<ForkOrigin>,
}

impl Default for ThreadViewStatus {
    fn default() -> Self {
        Self {
            schema: VIEW_SCHEMA,
            sequence: 0,
            turn: TurnState::Idle,
            last_outcome: None,
            open_messages: Vec::new(),
            open_tools: Vec::new(),
            recently_closed: Vec::new(),
            pending_confirmations: Vec::new(),
            pending_questions: Vec::new(),
            retry: None,
            usage: UsageView::default(),
            stalled: false,
            forked_from: None,
        }
    }
}

/// The source of a forked thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkOrigin {
    pub source_thread_id: ThreadId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_committed_turns: Option<u32>,
}

/// Whether a text item is model output or model thinking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextKind {
    Text,
    Thinking,
}

/// The lifecycle of a tool call item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolState {
    Running,
    AwaitingConfirmation,
    Succeeded,
    Failed,
    Interrupted,
}

/// The latest progress report of a tool call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolProgressView {
    pub stage: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// A confirmation prompt on a tool call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmationView {
    pub description: String,
    pub input: serde_json::Value,
}

/// The latest progress report of a subagent spawned by a tool call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentView {
    pub subagent_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nickname: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child_thread_id: Option<ThreadId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child_root_task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_turn: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub tool_name: String,
    pub tool_context: String,
    pub completed: bool,
    pub success: bool,
    pub tool_count: u32,
    pub total_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

/// A tool call, its confirmation prompt and its subagent, if any.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallView {
    pub id: String,
    pub name: String,
    pub display_name: String,
    #[serde(default)]
    pub input: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<ToolTier>,
    pub state: ToolState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<ToolResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<ToolProgressView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmation: Option<ConfirmationView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent: Option<SubagentView>,
    pub relayed: bool,
}

impl ToolCallView {
    fn new(id: &str, name: &str, display_name: &str, state: ToolState) -> Self {
        Self {
            id: id.to_owned(),
            name: name.to_owned(),
            display_name: display_name.to_owned(),
            input: serde_json::Value::Null,
            tier: None,
            state,
            result: None,
            progress: None,
            confirmation: None,
            subagent: None,
            relayed: false,
        }
    }

    fn into_item(self) -> ViewItem {
        ViewItem::ToolCall(Box::new(self))
    }
}

/// One entry of the timeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewItem {
    UserInput {
        content: Vec<ContentBlock>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emitter_task_id: Option<String>,
    },
    Text {
        text_kind: TextKind,
        message_id: String,
        text: String,
        open: bool,
        interrupted: bool,
    },
    ToolCall(Box<ToolCallView>),
    Question {
        task_id: String,
        question: QuestionPayload,
        resolved: bool,
    },
    Error {
        message: String,
        recoverable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<TerminalReason>,
    },
    Refusal {
        message_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    Compaction {
        original_count: usize,
        new_count: usize,
        original_tokens: usize,
        new_tokens: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trigger: Option<CompactionTrigger>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        engine: Option<CompactionEngine>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        occupancy: Option<CompactionOccupancy>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        threshold: Option<CompactionThreshold>,
    },
    AccountRotation {
        provider: String,
        from_account: String,
        to_account: String,
        reason: AccountRotationReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_seconds: Option<u32>,
    },
    AccountPoolExhausted {
        provider: String,
        account_count: u32,
        reason: AccountRotationReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_seconds: Option<u32>,
    },
    TurnEnd {
        turn: usize,
        outcome: TurnOutcome,
        usage: TokenUsage,
    },
}

/// A change to one item.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "patch", rename_all = "snake_case")]
pub enum ItemPatch {
    /// Close a text item. `text` replaces the streamed text when the final
    /// event carries it.
    CloseText {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        interrupted: bool,
    },
    ToolEnded {
        result: ToolResult,
    },
    ToolInterrupted,
    ToolProgress {
        progress: ToolProgressView,
    },
    AwaitingConfirmation {
        confirmation: ConfirmationView,
    },
    Subagent {
        subagent: SubagentView,
    },
    QuestionResolved,
}

/// One change to the view. The ops of one event, applied in order, turn the
/// view before the event into the view after it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ViewOp {
    SetStatus {
        status: Box<ThreadViewStatus>,
    },
    /// Replace the body of the item with `key`, keeping its position, or
    /// append a new item at the end of the timeline.
    Put {
        key: String,
        item: ViewItem,
    },
    AppendText {
        key: String,
        delta: String,
    },
    Patch {
        key: String,
        patch: ItemPatch,
    },
}

/// Fold one committed event into the view.
///
/// Returns the next status and the ops that produce it; the last op is
/// always [`ViewOp::SetStatus`]. Pure: the same inputs give the same output.
#[must_use]
pub fn fold(
    status: &ThreadViewStatus,
    sequence: u64,
    at: OffsetDateTime,
    event: &AgentEvent,
    config: &ViewConfig,
) -> (ThreadViewStatus, Vec<ViewOp>) {
    let mut step = Step {
        status: status.clone(),
        ops: Vec::new(),
        sequence,
    };
    step.status.schema = VIEW_SCHEMA;
    step.status.sequence = sequence;
    step.status.stalled = false;
    step.apply(at, event, config);
    let status = step.status;
    let mut ops = step.ops;
    ops.push(ViewOp::SetStatus {
        status: Box::new(status.clone()),
    });
    (status, ops)
}

struct Step {
    status: ThreadViewStatus,
    ops: Vec<ViewOp>,
    sequence: u64,
}

fn message_key(message_id: &str) -> String {
    format!("msg:{message_id}")
}

fn tool_key(id: &str) -> String {
    format!("tool:{id}")
}

fn question_key(tool_call_id: &str) -> String {
    format!("question:{tool_call_id}")
}

fn run_end(event: &AgentEvent) -> Option<(usize, TokenUsage, TurnOutcome)> {
    if let AgentEvent::Done {
        total_turns,
        total_usage,
        estimated_cost_usd,
        ..
    } = event
    {
        return Some((
            *total_turns,
            total_usage.clone(),
            TurnOutcome::Done {
                estimated_cost_usd: *estimated_cost_usd,
            },
        ));
    }
    if let AgentEvent::BudgetExceeded {
        total_turns,
        total_usage,
        limit,
        ..
    } = event
    {
        return Some((
            *total_turns,
            total_usage.clone(),
            TurnOutcome::BudgetExceeded { limit: *limit },
        ));
    }
    let AgentEvent::Cancelled {
        turn,
        usage,
        reason,
        ..
    } = event
    else {
        return None;
    };
    Some((
        *turn,
        usage.clone(),
        TurnOutcome::Cancelled {
            reason: reason.clone(),
        },
    ))
}

fn notice_item(event: &AgentEvent) -> Option<ViewItem> {
    if let AgentEvent::AccountRotation {
        provider,
        from_account,
        to_account,
        reason,
        retry_after_seconds,
    } = event
    {
        return Some(ViewItem::AccountRotation {
            provider: provider.clone(),
            from_account: from_account.clone(),
            to_account: to_account.clone(),
            reason: *reason,
            retry_after_seconds: *retry_after_seconds,
        });
    }
    if let AgentEvent::AccountPoolExhausted {
        provider,
        account_count,
        reason,
        retry_after_seconds,
    } = event
    {
        return Some(ViewItem::AccountPoolExhausted {
            provider: provider.clone(),
            account_count: *account_count,
            reason: *reason,
            retry_after_seconds: *retry_after_seconds,
        });
    }
    let AgentEvent::ContextCompacted {
        original_count,
        new_count,
        original_tokens,
        new_tokens,
        trigger,
        engine,
        occupancy,
        threshold,
    } = event
    else {
        return None;
    };
    Some(ViewItem::Compaction {
        original_count: *original_count,
        new_count: *new_count,
        original_tokens: *original_tokens,
        new_tokens: *new_tokens,
        trigger: *trigger,
        engine: *engine,
        occupancy: *occupancy,
        threshold: *threshold,
    })
}

fn subagent_view(event: &AgentEvent) -> Option<SubagentView> {
    let AgentEvent::SubagentProgress {
        subagent_name,
        nickname,
        child_thread_id,
        child_root_task_id,
        subagent_task_id,
        max_turns,
        current_turn,
        model,
        tool_name,
        tool_context,
        completed,
        success,
        tool_count,
        total_tokens,
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
        ..
    } = event
    else {
        return None;
    };
    Some(SubagentView {
        subagent_name: subagent_name.clone(),
        nickname: nickname.clone(),
        child_thread_id: child_thread_id.clone(),
        child_root_task_id: child_root_task_id.clone(),
        subagent_task_id: subagent_task_id.clone(),
        max_turns: *max_turns,
        current_turn: *current_turn,
        model: model.clone(),
        tool_name: tool_name.clone(),
        tool_context: tool_context.clone(),
        completed: *completed,
        success: *success,
        tool_count: *tool_count,
        total_tokens: *total_tokens,
        input_tokens: *input_tokens,
        output_tokens: *output_tokens,
        cache_read_input_tokens: *cache_read_input_tokens,
        cache_creation_input_tokens: *cache_creation_input_tokens,
    })
}

impl Step {
    fn apply(&mut self, at: OffsetDateTime, event: &AgentEvent, config: &ViewConfig) {
        match event {
            AgentEvent::ThreadCreated {
                source_thread_id,
                fork_after_committed_turns,
                ..
            } => {
                self.status.forked_from = source_thread_id.clone().map(|source| ForkOrigin {
                    source_thread_id: source,
                    after_committed_turns: *fork_after_committed_turns,
                });
            }
            AgentEvent::Start {
                turn,
                emitter_task_id,
                ..
            } => self.start(*turn, emitter_task_id.clone()),
            AgentEvent::TurnComplete { turn, usage, .. } => {
                self.status.usage.last_turn = usage.clone();
                self.status.usage.turns = *turn;
                self.status.retry = None;
            }
            AgentEvent::Done { .. }
            | AgentEvent::BudgetExceeded { .. }
            | AgentEvent::Cancelled { .. } => {
                if let Some((turns, usage, outcome)) = run_end(event) {
                    self.end_turn(turns, &usage, outcome);
                }
            }
            AgentEvent::Error {
                message,
                recoverable,
                reason,
                ..
            } => self.error(message, *recoverable, reason.as_ref()),
            AgentEvent::AutoRetryStart {
                attempt,
                max_attempts,
                delay_ms,
                error_message,
            } => {
                self.status.retry = Some(RetryState {
                    attempt: *attempt,
                    max_attempts: *max_attempts,
                    delay_ms: *delay_ms,
                    error_message: error_message.clone(),
                    started_at: at,
                });
            }
            AgentEvent::AutoRetryEnd { .. } => self.status.retry = None,
            AgentEvent::UserInput {
                content,
                emitter_task_id,
                ..
            } => self.put(
                format!("user:{}", self.sequence),
                ViewItem::UserInput {
                    content: content.clone(),
                    emitter_task_id: emitter_task_id.clone(),
                },
            ),
            AgentEvent::Refusal { message_id, text } => self.put_closed(
                format!("refusal:{message_id}"),
                ViewItem::Refusal {
                    message_id: message_id.clone(),
                    text: text.clone(),
                },
            ),
            AgentEvent::Thinking { .. }
            | AgentEvent::ThinkingDelta { .. }
            | AgentEvent::TextDelta { .. }
            | AgentEvent::Text { .. }
            | AgentEvent::ToolCallStart { .. }
            | AgentEvent::ToolCallEnd { .. }
            | AgentEvent::ToolProgress { .. }
            | AgentEvent::ToolRequiresConfirmation { .. }
            | AgentEvent::QuestionAsked { .. }
            | AgentEvent::AccountRotation { .. }
            | AgentEvent::AccountPoolExhausted { .. }
            | AgentEvent::ContextCompacted { .. }
            | AgentEvent::SubagentProgress { .. } => self.apply_content(event, config),
        }
    }

    fn apply_content(&mut self, event: &AgentEvent, config: &ViewConfig) {
        match event {
            AgentEvent::Thinking {
                message_id,
                text,
                emitter_task_id,
            } => self.final_text(
                TextKind::Thinking,
                message_id,
                text,
                emitter_task_id.as_deref(),
            ),
            AgentEvent::Text {
                message_id,
                text,
                emitter_task_id,
            } => self.final_text(TextKind::Text, message_id, text, emitter_task_id.as_deref()),
            AgentEvent::ThinkingDelta {
                message_id,
                delta,
                emitter_task_id,
            } => self.delta(
                TextKind::Thinking,
                message_id,
                delta,
                emitter_task_id.as_deref(),
            ),
            AgentEvent::TextDelta {
                message_id,
                delta,
                emitter_task_id,
            } => self.delta(
                TextKind::Text,
                message_id,
                delta,
                emitter_task_id.as_deref(),
            ),
            AgentEvent::ToolCallStart {
                id,
                name,
                display_name,
                input,
                tier,
            } => self.tool_start(id, name, display_name, input, *tier),
            AgentEvent::ToolCallEnd {
                id,
                name,
                display_name,
                result,
            } => self.tool_end(id, name, display_name, result),
            AgentEvent::ToolProgress {
                id,
                name,
                stage,
                message,
                data,
                ..
            } => {
                if config.stall_tool_name.as_deref() == Some(name.as_str()) {
                    self.status.stalled = true;
                } else {
                    self.tool_progress(id, stage, message, data.as_ref());
                }
            }
            AgentEvent::ToolRequiresConfirmation {
                id,
                name,
                display_name,
                input,
                description,
            } => self.confirmation(id, name, display_name, input, description),
            AgentEvent::QuestionAsked { task_id, questions } => {
                self.questions(task_id, questions);
            }
            AgentEvent::AccountRotation { .. }
            | AgentEvent::AccountPoolExhausted { .. }
            | AgentEvent::ContextCompacted { .. } => {
                if let Some(item) = notice_item(event) {
                    self.put(format!("notice:{}", self.sequence), item);
                }
            }
            AgentEvent::SubagentProgress { subagent_id, .. } => {
                if let Some(subagent) = subagent_view(event) {
                    self.subagent(subagent_id, subagent);
                }
            }
            AgentEvent::ThreadCreated { .. }
            | AgentEvent::UserInput { .. }
            | AgentEvent::Refusal { .. }
            | AgentEvent::Start { .. }
            | AgentEvent::TurnComplete { .. }
            | AgentEvent::Done { .. }
            | AgentEvent::BudgetExceeded { .. }
            | AgentEvent::Error { .. }
            | AgentEvent::AutoRetryStart { .. }
            | AgentEvent::AutoRetryEnd { .. }
            | AgentEvent::Cancelled { .. } => {}
        }
    }

    fn start(&mut self, turn: usize, task_id: Option<String>) {
        self.interrupt_open();
        self.resolve_turn_prompts();
        self.status.turn = TurnState::Running { turn, task_id };
        self.status.last_outcome = None;
    }

    fn tool_start(
        &mut self,
        id: &str,
        name: &str,
        display_name: &str,
        input: &serde_json::Value,
        tier: ToolTier,
    ) {
        let key = tool_key(id);
        let item = ToolCallView {
            input: input.clone(),
            tier: Some(tier),
            ..ToolCallView::new(id, name, display_name, ToolState::Running)
        }
        .into_item();
        self.put(key.clone(), item);
        if !self.is_open_tool(&key) {
            self.forget_closed(&key);
            self.status.open_tools.push(key);
        }
    }

    fn tool_progress(
        &mut self,
        id: &str,
        stage: &str,
        message: &str,
        data: Option<&serde_json::Value>,
    ) {
        let key = tool_key(id);
        if self.is_open_tool(&key) {
            self.patch(
                key,
                ItemPatch::ToolProgress {
                    progress: ToolProgressView {
                        stage: stage.to_owned(),
                        message: message.to_owned(),
                        data: data.cloned(),
                    },
                },
            );
        }
    }

    fn confirmation(
        &mut self,
        id: &str,
        name: &str,
        display_name: &str,
        input: &serde_json::Value,
        description: &str,
    ) {
        let key = tool_key(id);
        let confirmation = ConfirmationView {
            description: description.to_owned(),
            input: input.clone(),
        };
        let relayed = !self.is_open_tool(&key);
        if relayed {
            let item = ToolCallView {
                input: input.clone(),
                confirmation: Some(confirmation),
                relayed: true,
                ..ToolCallView::new(id, name, display_name, ToolState::AwaitingConfirmation)
            }
            .into_item();
            self.put(key.clone(), item);
        } else {
            self.patch(
                key.clone(),
                ItemPatch::AwaitingConfirmation { confirmation },
            );
        }
        self.status.pending_confirmations.retain(|p| p.key != key);
        self.status
            .pending_confirmations
            .push(PendingConfirmation { key, relayed });
    }

    fn questions(&mut self, task_id: &str, questions: &[QuestionPayload]) {
        for question in questions {
            let key = question_key(&question.tool_call_id);
            if self.status.pending_questions.contains(&key) {
                continue;
            }
            self.put_closed(
                key.clone(),
                ViewItem::Question {
                    task_id: task_id.to_owned(),
                    question: question.clone(),
                    resolved: false,
                },
            );
            self.status.pending_questions.push(key);
        }
    }

    fn error(&mut self, message: &str, recoverable: bool, reason: Option<&TerminalReason>) {
        self.put(
            format!("error:{}", self.sequence),
            ViewItem::Error {
                message: message.to_owned(),
                recoverable,
                reason: reason.cloned(),
            },
        );
        if let Some(reason) = reason {
            let turns = self.status.usage.turns;
            let usage = self.status.usage.last_turn.clone();
            self.end_turn(
                turns,
                &usage,
                TurnOutcome::Failed {
                    message: message.to_owned(),
                    reason: reason.clone(),
                },
            );
        }
    }

    fn subagent(&mut self, subagent_id: &str, subagent: SubagentView) {
        let key = tool_key(subagent_id);
        if self.is_known_tool(&key) {
            self.patch(key, ItemPatch::Subagent { subagent });
            return;
        }
        let state = match (subagent.completed, subagent.success) {
            (false, _) => ToolState::Running,
            (true, true) => ToolState::Succeeded,
            (true, false) => ToolState::Failed,
        };
        let name = subagent.subagent_name.clone();
        self.put_closed(
            key,
            ToolCallView {
                subagent: Some(subagent),
                ..ToolCallView::new(subagent_id, &name, &name, state)
            }
            .into_item(),
        );
    }

    fn put(&mut self, key: String, item: ViewItem) {
        self.ops.push(ViewOp::Put { key, item });
    }

    fn put_closed(&mut self, key: String, item: ViewItem) {
        self.put(key.clone(), item);
        self.remember_closed(key);
    }

    fn is_known_tool(&self, key: &str) -> bool {
        self.is_open_tool(key)
            || self.is_pending_confirmation(key)
            || self.status.recently_closed.iter().any(|k| k == key)
    }

    fn is_pending_confirmation(&self, key: &str) -> bool {
        self.status
            .pending_confirmations
            .iter()
            .any(|p| p.key == key)
    }

    fn patch(&mut self, key: String, patch: ItemPatch) {
        self.ops.push(ViewOp::Patch { key, patch });
    }

    fn is_open_tool(&self, key: &str) -> bool {
        self.status.open_tools.iter().any(|k| k == key)
    }

    fn is_open_message(&self, key: &str) -> bool {
        self.status.open_messages.iter().any(|k| k == key)
    }

    fn remember_closed(&mut self, key: String) {
        self.status.recently_closed.retain(|k| *k != key);
        self.status.recently_closed.push(key);
        let excess = self
            .status
            .recently_closed
            .len()
            .saturating_sub(RECENTLY_CLOSED_LIMIT);
        self.status.recently_closed.drain(..excess);
    }

    fn forget_closed(&mut self, key: &str) {
        self.status.recently_closed.retain(|k| k != key);
    }

    fn is_stale(&self, emitter_task_id: Option<&str>) -> bool {
        match (&self.status.turn, emitter_task_id) {
            (
                TurnState::Running {
                    task_id: Some(current),
                    ..
                },
                Some(emitter),
            ) => current != emitter,
            (TurnState::Running { .. }, _) => false,
            (TurnState::Idle, _) => true,
        }
    }

    fn delta(&mut self, kind: TextKind, message_id: &str, delta: &str, emitter: Option<&str>) {
        let key = message_key(message_id);
        if self.is_open_message(&key) {
            self.ops.push(ViewOp::AppendText {
                key,
                delta: delta.to_owned(),
            });
            return;
        }
        if self.is_stale(emitter) || self.status.recently_closed.contains(&key) {
            return;
        }
        self.put(
            key.clone(),
            ViewItem::Text {
                text_kind: kind,
                message_id: message_id.to_owned(),
                text: delta.to_owned(),
                open: true,
                interrupted: false,
            },
        );
        self.status.open_messages.push(key);
    }

    fn final_text(&mut self, kind: TextKind, message_id: &str, text: &str, emitter: Option<&str>) {
        let key = message_key(message_id);
        let stale = self.is_stale(emitter);
        if self.is_open_message(&key) {
            self.status.open_messages.retain(|k| *k != key);
            self.patch(
                key.clone(),
                ItemPatch::CloseText {
                    text: Some(text.to_owned()),
                    interrupted: false,
                },
            );
        } else if self.status.recently_closed.contains(&key) {
            self.patch(
                key.clone(),
                ItemPatch::CloseText {
                    text: Some(text.to_owned()),
                    interrupted: stale,
                },
            );
        } else {
            self.put(
                key.clone(),
                ViewItem::Text {
                    text_kind: kind,
                    message_id: message_id.to_owned(),
                    text: text.to_owned(),
                    open: false,
                    interrupted: stale,
                },
            );
        }
        self.remember_closed(key);
    }

    fn tool_end(&mut self, id: &str, name: &str, display_name: &str, result: &ToolResult) {
        let key = tool_key(id);
        let question = question_key(id);
        if self.status.pending_questions.contains(&question) {
            self.status.pending_questions.retain(|k| *k != question);
            self.patch(question, ItemPatch::QuestionResolved);
        }
        let known = self.is_known_tool(&key);
        self.status.pending_confirmations.retain(|p| p.key != key);
        self.status.open_tools.retain(|k| *k != key);
        if known {
            self.patch(
                key.clone(),
                ItemPatch::ToolEnded {
                    result: result.clone(),
                },
            );
        } else {
            let state = if result.success {
                ToolState::Succeeded
            } else {
                ToolState::Failed
            };
            self.put(
                key.clone(),
                ToolCallView {
                    result: Some(result.clone()),
                    ..ToolCallView::new(id, name, display_name, state)
                }
                .into_item(),
            );
        }
        self.remember_closed(key);
    }

    fn interrupt_open(&mut self) {
        for key in std::mem::take(&mut self.status.open_messages) {
            self.patch(
                key.clone(),
                ItemPatch::CloseText {
                    text: None,
                    interrupted: true,
                },
            );
            self.remember_closed(key);
        }
        for key in std::mem::take(&mut self.status.open_tools) {
            self.patch(key.clone(), ItemPatch::ToolInterrupted);
            self.remember_closed(key);
        }
    }

    fn resolve_turn_prompts(&mut self) {
        self.status.pending_confirmations.retain(|p| p.relayed);
        for key in std::mem::take(&mut self.status.pending_questions) {
            self.patch(key, ItemPatch::QuestionResolved);
        }
        self.status.retry = None;
    }

    fn end_turn(&mut self, turns: usize, usage: &TokenUsage, outcome: TurnOutcome) {
        self.interrupt_open();
        self.resolve_turn_prompts();
        self.status.usage.run = usage.clone();
        self.status.usage.turns = turns;
        self.put(
            format!("turn_end:{}", self.sequence),
            ViewItem::TurnEnd {
                turn: turns,
                outcome: outcome.clone(),
                usage: usage.clone(),
            },
        );
        self.status.turn = TurnState::Idle;
        self.status.last_outcome = Some(outcome);
    }
}

/// A whole view in memory: the reference applier for [`ViewOp`]s.
///
/// The in-memory store keeps one per thread, tests compare it with the SQL
/// backends, and a client can keep one from a snapshot.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThreadView {
    pub status: ThreadViewStatus,
    pub items: BTreeMap<String, IndexedItem>,
}

/// An item and its position in the timeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexedItem {
    pub index: u64,
    pub item: ViewItem,
}

impl ThreadView {
    /// Fold events in sequence order into a new view.
    #[must_use]
    pub fn from_events<'a>(
        events: impl IntoIterator<Item = (u64, OffsetDateTime, &'a AgentEvent)>,
        config: &ViewConfig,
    ) -> Self {
        let mut view = Self::default();
        for (sequence, at, event) in events {
            view.fold(sequence, at, event, config);
        }
        view
    }

    /// Fold one event and apply its ops. Returns the ops.
    pub fn fold(
        &mut self,
        sequence: u64,
        at: OffsetDateTime,
        event: &AgentEvent,
        config: &ViewConfig,
    ) -> Vec<ViewOp> {
        let (_, ops) = fold(&self.status, sequence, at, event, config);
        for op in &ops {
            self.apply(op);
        }
        ops
    }

    /// Apply one op. Returns false, and changes nothing, for a patch or text
    /// append that addresses an item the view does not have: an item from
    /// before the view's seed or retention floor.
    pub fn apply(&mut self, op: &ViewOp) -> bool {
        match op {
            ViewOp::SetStatus { status } => self.status = (**status).clone(),
            ViewOp::Put { key, item } => {
                let index = self.items.len() as u64;
                self.items
                    .entry(key.clone())
                    .and_modify(|existing| existing.item = item.clone())
                    .or_insert_with(|| IndexedItem {
                        index,
                        item: item.clone(),
                    });
            }
            ViewOp::AppendText { key, delta } => {
                let Some(entry) = self.items.get_mut(key) else {
                    return false;
                };
                if let ViewItem::Text { text, .. } = &mut entry.item {
                    text.push_str(delta);
                }
            }
            ViewOp::Patch { key, patch } => {
                let Some(entry) = self.items.get_mut(key) else {
                    return false;
                };
                apply_patch(&mut entry.item, patch);
            }
        }
        true
    }

    /// The items in timeline order.
    pub fn timeline(&self) -> impl Iterator<Item = (&str, &ViewItem)> {
        let mut items: Vec<_> = self.items.iter().collect();
        items.sort_by_key(|(_, indexed)| indexed.index);
        items
            .into_iter()
            .map(|(key, indexed)| (key.as_str(), &indexed.item))
    }
}

fn tool_patch(tool: &mut ToolCallView, patch: &ItemPatch) {
    match patch {
        ItemPatch::ToolEnded { result } => {
            tool.state = if result.success {
                ToolState::Succeeded
            } else {
                ToolState::Failed
            };
            tool.result = Some(result.clone());
        }
        ItemPatch::ToolInterrupted => tool.state = ToolState::Interrupted,
        ItemPatch::ToolProgress { progress } => tool.progress = Some(progress.clone()),
        ItemPatch::AwaitingConfirmation { confirmation } => {
            tool.state = ToolState::AwaitingConfirmation;
            tool.confirmation = Some(confirmation.clone());
        }
        ItemPatch::Subagent { subagent } => tool.subagent = Some(subagent.clone()),
        ItemPatch::CloseText { .. } | ItemPatch::QuestionResolved => {}
    }
}

/// Apply a patch to an item body. A patch for another kind of item leaves it
/// unchanged.
pub fn apply_patch(item: &mut ViewItem, patch: &ItemPatch) {
    match (item, patch) {
        (
            ViewItem::Text {
                text,
                open,
                interrupted,
                ..
            },
            ItemPatch::CloseText {
                text: final_text,
                interrupted: was_interrupted,
            },
        ) => {
            if let Some(final_text) = final_text {
                text.clone_from(final_text);
            }
            *open = false;
            *interrupted = *was_interrupted;
        }
        (ViewItem::ToolCall(tool), patch) => tool_patch(tool, patch),
        (ViewItem::Question { resolved, .. }, ItemPatch::QuestionResolved) => *resolved = true,
        (
            ViewItem::UserInput { .. }
            | ViewItem::Text { .. }
            | ViewItem::Question { .. }
            | ViewItem::Error { .. }
            | ViewItem::Refusal { .. }
            | ViewItem::Compaction { .. }
            | ViewItem::AccountRotation { .. }
            | ViewItem::AccountPoolExhausted { .. }
            | ViewItem::TurnEnd { .. },
            ItemPatch::CloseText { .. }
            | ItemPatch::ToolEnded { .. }
            | ItemPatch::ToolInterrupted
            | ItemPatch::ToolProgress { .. }
            | ItemPatch::AwaitingConfirmation { .. }
            | ItemPatch::Subagent { .. }
            | ItemPatch::QuestionResolved,
        ) => {}
    }
}

#[cfg(test)]
#[path = "view_test.rs"]
mod tests;
