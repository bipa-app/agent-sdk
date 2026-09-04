//! Maps durable SDK events into ACP `session/update` payloads.
//!
//! Subagent lifecycle (design §3.1, host-path semantics only):
//!
//! 1. `ToolCallStart` opens the row; its id IS the later
//!    `SubagentProgress.subagent_id`, so no correlation table exists.
//! 2. No `ToolCallEnd` is ever emitted for a subagent invocation. The row
//!    closes ONLY on `SubagentProgress { completed: true }`, with the
//!    status taken from `success`. Progress ticks map to `in_progress`
//!    updates carrying a turn/tool/token summary.
//! 3. The in-process SDK loop emits the same variant per CHILD tool call;
//!    this mapper is correct on the durable host path, where the variant
//!    means the whole invocation.
//! 4. Result text never rides the event stream. A close is handed to the
//!    run loop as [`Mapped::SubagentClosed`] so it can read the
//!    invocation's result from the backend before emitting the terminal
//!    update; the tick summary is the fallback content.
//!
//! Plan synthesis: `todo_write` input is cached at `ToolCallStart` (it is
//! unvalidated there) and turned into an ACP `plan` update only after the
//! matching SUCCESSFUL `ToolCallEnd`. A failed `todo_write` releases the
//! cache and surfaces no plan.

use std::collections::{HashMap, HashSet};

use agent_sdk_foundation::{AgentEvent, TokenUsage};
use serde_json::{Value, json};

use crate::wire::StopReason;

/// Outcome of mapping one committed event.
#[derive(Debug, PartialEq, Eq)]
pub enum Mapped {
    /// Emit these `session/update` payloads, in order. Never empty.
    Update(Vec<Value>),
    /// A subagent invocation closed; the run loop reads its result text
    /// from the backend, then emits [`SubagentClosed::update`].
    SubagentClosed(SubagentClosed),
    /// The turn is over with this stop reason.
    Terminal(StopReason),
    /// The turn failed; resolve the prompt as a JSON-RPC error.
    Fail(String),
    /// This event has no ACP representation in this slice.
    Ignore,
}

/// A subagent invocation's terminal state, minus the result text that
/// only the backend's task store holds (§3.1 rule 4).
#[derive(Debug, PartialEq, Eq)]
pub struct SubagentClosed {
    /// The spawning tool call — the row this close resolves.
    pub tool_call_id: String,
    /// Durable invocation task to read the result from, when the event
    /// carried one.
    pub subagent_task_id: Option<String>,
    /// Whether the invocation succeeded.
    pub success: bool,
    /// Turn/tool/token summary: the terminal content when no result text
    /// is available.
    pub summary: String,
}

impl SubagentClosed {
    /// The terminal `tool_call_update`, with `result_text` as content when
    /// the backend supplied it and the progress summary otherwise.
    #[must_use]
    pub fn update(&self, result_text: Option<&str>) -> Value {
        tool_call_end(
            &self.tool_call_id,
            self.success,
            result_text.unwrap_or(&self.summary),
        )
    }
}

/// Per-prompt mapper state.
#[derive(Debug, Default)]
pub struct EventMapper {
    /// Message ids whose text arrived as deltas; their consolidated
    /// `Text` is suppressed.
    text_delta_seen: HashSet<String>,
    /// Same for thinking.
    thinking_delta_seen: HashSet<String>,
    /// `todo_write` inputs cached at `ToolCallStart`, keyed by tool call
    /// id, released at the matching `ToolCallEnd`.
    pending_todo_writes: HashMap<String, Value>,
    /// Subagent ids already closed. A completion commit is retried until
    /// it lands, so a second `completed: true` for the same id must not
    /// re-close (or re-tick) the row.
    closed_subagents: HashSet<String>,
}

impl EventMapper {
    pub fn map(&mut self, event: &AgentEvent) -> Mapped {
        match event {
            AgentEvent::TextDelta {
                message_id, delta, ..
            } => {
                self.text_delta_seen.insert(message_id.clone());
                Mapped::Update(vec![text_chunk("agent_message_chunk", delta)])
            }
            AgentEvent::Text {
                message_id, text, ..
            } => map_consolidated(
                &self.text_delta_seen,
                message_id,
                text,
                "agent_message_chunk",
            ),
            AgentEvent::ThinkingDelta {
                message_id, delta, ..
            } => {
                self.thinking_delta_seen.insert(message_id.clone());
                Mapped::Update(vec![text_chunk("agent_thought_chunk", delta)])
            }
            AgentEvent::Thinking {
                message_id, text, ..
            } => map_consolidated(
                &self.thinking_delta_seen,
                message_id,
                text,
                "agent_thought_chunk",
            ),
            AgentEvent::ToolCallStart {
                id,
                name,
                display_name,
                input,
                ..
            } => {
                if name == TODO_WRITE_TOOL {
                    self.pending_todo_writes.insert(id.clone(), input.clone());
                }
                Mapped::Update(vec![tool_call_start(id, name, display_name, input)])
            }
            AgentEvent::ToolProgress { id, message, .. } => {
                Mapped::Update(vec![tool_progress(id, message)])
            }
            AgentEvent::ToolCallEnd { id, result, .. } => {
                let mut updates = vec![tool_call_end(id, result.success, &result.output)];
                let plan = self
                    .pending_todo_writes
                    .remove(id)
                    .filter(|_| result.success)
                    .and_then(|input| synthesize_plan(&input));
                updates.extend(plan);
                Mapped::Update(updates)
            }
            AgentEvent::SubagentProgress {
                subagent_id,
                subagent_name,
                nickname,
                subagent_task_id,
                max_turns,
                current_turn,
                completed,
                success,
                tool_count,
                total_tokens,
                ..
            } => {
                if self.closed_subagents.contains(subagent_id) {
                    return Mapped::Ignore;
                }
                let summary = subagent_summary(
                    subagent_name,
                    nickname.as_deref(),
                    *current_turn,
                    *max_turns,
                    *tool_count,
                    *total_tokens,
                );
                if !*completed {
                    return Mapped::Update(vec![tool_progress(subagent_id, &summary)]);
                }
                self.closed_subagents.insert(subagent_id.clone());
                Mapped::SubagentClosed(SubagentClosed {
                    tool_call_id: subagent_id.clone(),
                    subagent_task_id: subagent_task_id.clone(),
                    success: *success,
                    summary,
                })
            }
            AgentEvent::TurnComplete { turn, usage, .. } => {
                Mapped::Update(vec![usage_update(*turn, usage)])
            }
            AgentEvent::Done { .. } => Mapped::Terminal(StopReason::EndTurn),
            AgentEvent::Cancelled { .. } => Mapped::Terminal(StopReason::Cancelled),
            AgentEvent::Refusal { .. } => Mapped::Terminal(StopReason::Refusal),
            AgentEvent::BudgetExceeded { .. } => Mapped::Terminal(StopReason::MaxTokens),
            AgentEvent::Error { message, .. } => Mapped::Fail(message.clone()),
            _ => Mapped::Ignore,
        }
    }
}

/// The SDK's plan-tracking tool; its successful writes become ACP plans.
const TODO_WRITE_TOOL: &str = "todo_write";

/// ACP plan entries carry a priority; `todo_write` has no such notion, so
/// every synthesized entry gets the neutral one.
const PLAN_ENTRY_PRIORITY: &str = "medium";

/// Build the ACP `plan` update from a validated `todo_write` input
/// (`{ todos: [{ content, status, activeForm }] }`). Statuses are the
/// same vocabulary on both sides (`pending`/`in_progress`/`completed`);
/// `activeForm` has no ACP counterpart. `None` when the input has no
/// `todos` array — the tool accepted it, so this is defensive, not
/// expected.
fn synthesize_plan(input: &Value) -> Option<Value> {
    let todos = input.get("todos")?.as_array()?;
    let entries: Vec<Value> = todos
        .iter()
        .filter_map(|todo| {
            let content = todo.get("content")?.as_str()?;
            let status = todo.get("status")?.as_str()?;
            Some(json!({
                "content": content,
                "priority": PLAN_ENTRY_PRIORITY,
                "status": status,
            }))
        })
        .collect();
    Some(json!({ "sessionUpdate": "plan", "entries": entries }))
}

/// One line of subagent progress for the activity feed, e.g.
/// `Zara (explore) · turn 2/10 · 3 tools · 1234 tokens`.
fn subagent_summary(
    name: &str,
    nickname: Option<&str>,
    current_turn: Option<u32>,
    max_turns: Option<u32>,
    tool_count: u32,
    total_tokens: u64,
) -> String {
    let label = nickname.map_or_else(
        || name.to_owned(),
        |nickname| format!("{nickname} ({name})"),
    );
    let turn = match (current_turn, max_turns) {
        (Some(turn), Some(max)) => format!(" · turn {turn}/{max}"),
        (Some(turn), None) => format!(" · turn {turn}"),
        (None, _) => String::new(),
    };
    let tools = if tool_count == 1 { "tool" } else { "tools" };
    format!("{label}{turn} · {tool_count} {tools} · {total_tokens} tokens")
}

fn map_consolidated(
    delta_seen: &HashSet<String>,
    message_id: &str,
    text: &str,
    update_kind: &str,
) -> Mapped {
    if delta_seen.contains(message_id) {
        Mapped::Ignore
    } else {
        Mapped::Update(vec![text_chunk(update_kind, text)])
    }
}

fn text_chunk(update_kind: &str, text: &str) -> Value {
    json!({
        "sessionUpdate": update_kind,
        "content": { "type": "text", "text": text },
    })
}

fn tool_call_start(id: &str, name: &str, display_name: &str, input: &Value) -> Value {
    json!({
        "sessionUpdate": "tool_call",
        "toolCallId": id,
        "title": display_name,
        "kind": tool_kind(name),
        "status": "pending",
        "rawInput": input,
    })
}

fn tool_progress(id: &str, message: &str) -> Value {
    json!({
        "sessionUpdate": "tool_call_update",
        "toolCallId": id,
        "status": "in_progress",
        "content": [tool_output_content(message)],
    })
}

fn tool_call_end(id: &str, success: bool, output: &str) -> Value {
    let status = if success { "completed" } else { "failed" };
    json!({
        "sessionUpdate": "tool_call_update",
        "toolCallId": id,
        "status": status,
        "content": [tool_output_content(output)],
    })
}

fn tool_output_content(text: &str) -> Value {
    json!({
        "type": "content",
        "content": { "type": "text", "text": text },
    })
}

fn usage_update(turn: usize, usage: &TokenUsage) -> Value {
    json!({
        "sessionUpdate": "usage_update",
        "turn": turn,
        "inputTokens": usage.input_tokens,
        "outputTokens": usage.output_tokens,
        "cachedInputTokens": usage.cached_input_tokens,
        "cacheCreationInputTokens": usage.cache_creation_input_tokens,
    })
}

/// Tool-name → ACP kind. Two audiences share it: the SDK's built-ins and
/// every tool satoshi (the first production backend) registers. Kept as
/// data so additions are easy to audit. Mutating and managing tools are
/// deliberately absent — they, and anything else unknown (remote MCP
/// tools, future registrations), fall through to `other`, never a panic.
const TOOL_KIND_TABLE: &[(&str, &str)] = &[
    // SDK built-ins.
    ("read", "read"),
    ("notebook_read", "read"),
    ("glob", "search"),
    ("grep", "search"),
    ("bash", "execute"),
    ("write", "execute"),
    ("edit", "execute"),
    ("multi_edit", "execute"),
    ("notebook_edit", "execute"),
    ("link_fetch", "fetch"),
    ("todo_read", "think"),
    ("todo_write", "think"),
    ("ask_user", "think"),
    // Registered by both the SDK and satoshi.
    ("web_search", "search"),
    // Satoshi reads: logs, documents, schemas, graph and scratch state.
    ("analytics_describe_table", "read"),
    ("analytics_list_tables", "read"),
    ("check_integration_health", "read"),
    ("fraud_source_schema", "read"),
    ("get_document", "read"),
    ("get_employee_work", "read"),
    ("get_okr_status", "read"),
    ("get_rock_detail", "read"),
    ("get_skill_versions", "read"),
    ("get_team_alignment", "read"),
    ("graph_context_pack", "read"),
    ("graph_get", "read"),
    ("graph_neighbors", "read"),
    ("investigation_list", "read"),
    ("investigation_read", "read"),
    ("list_scheduled_tasks", "read"),
    ("list_skills", "read"),
    ("load_skill", "read"),
    ("parse_document", "read"),
    ("pg_describe_table", "read"),
    ("pg_list_tables", "read"),
    ("read_logs", "read"),
    ("read_pdf", "read"),
    ("scratch_read", "read"),
    // Satoshi searches and recalls.
    ("conversation_recall", "search"),
    ("deep_research", "search"),
    ("graph_search", "search"),
    ("investigation_search", "search"),
    ("knowledge_recall", "search"),
    ("search_knowledge", "search"),
    // Satoshi code and query execution.
    ("analytics_query", "execute"),
    ("execute_code", "execute"),
    ("fraud_execute_code", "execute"),
    ("fraud_source_query", "execute"),
    ("pg_query", "execute"),
    ("rlm", "execute"),
    ("sandbox_coding_agent", "execute"),
    // Satoshi fetching and deliberation.
    ("fetch_page", "fetch"),
    ("advisor", "think"),
];

fn tool_kind(name: &str) -> &'static str {
    for (known_name, kind) in TOOL_KIND_TABLE {
        if *known_name == name {
            return kind;
        }
    }
    "other"
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::{ThreadId, ToolResult, ToolTier};
    use std::time::Duration;

    fn done() -> AgentEvent {
        AgentEvent::Done {
            thread_id: ThreadId::from_string("t".to_owned()),
            total_turns: 1,
            total_usage: TokenUsage::default(),
            duration: Duration::from_millis(1),
            estimated_cost_usd: None,
            emitter_task_id: None,
        }
    }

    /// A durable-host `SubagentProgress` frame for subagent `id`.
    fn subagent_progress(id: &str, completed: bool, success: bool, turn: u32) -> AgentEvent {
        AgentEvent::SubagentProgress {
            subagent_id: id.to_owned(),
            subagent_name: "explore".to_owned(),
            nickname: Some("Zara".to_owned()),
            child_thread_id: Some(ThreadId::from_string("child".to_owned())),
            child_root_task_id: Some("child-root".to_owned()),
            subagent_task_id: Some("invocation-1".to_owned()),
            max_turns: Some(10),
            current_turn: Some(turn),
            model: Some("model".to_owned()),
            tool_name: "explore".to_owned(),
            tool_context: "look around".to_owned(),
            completed,
            success,
            tool_count: 3,
            total_tokens: 1234,
            input_tokens: 1000,
            output_tokens: 234,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
        }
    }

    fn tool_update(id: &str, status: &str, text: &str) -> Value {
        json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": id,
            "status": status,
            "content": [{
                "type": "content",
                "content": { "type": "text", "text": text },
            }],
        })
    }

    #[test]
    fn golden_text_delta_and_consolidated_dedupe() {
        let mut mapper = EventMapper::default();
        assert_eq!(
            mapper.map(&AgentEvent::text_delta("m1", "hel")),
            Mapped::Update(vec![json!({
                "sessionUpdate": "agent_message_chunk",
                "content": { "type": "text", "text": "hel" },
            })])
        );
        assert_eq!(mapper.map(&AgentEvent::text("m1", "hello")), Mapped::Ignore);
        assert_eq!(
            mapper.map(&AgentEvent::text("m2", "whole")),
            Mapped::Update(vec![json!({
                "sessionUpdate": "agent_message_chunk",
                "content": { "type": "text", "text": "whole" },
            })])
        );
    }

    #[test]
    fn golden_thinking_delta_and_consolidated_dedupe() {
        let mut mapper = EventMapper::default();
        assert_eq!(
            mapper.map(&AgentEvent::thinking_delta("m1", "hmm")),
            Mapped::Update(vec![json!({
                "sessionUpdate": "agent_thought_chunk",
                "content": { "type": "text", "text": "hmm" },
            })])
        );
        assert_eq!(
            mapper.map(&AgentEvent::thinking("m1", "hmm, yes")),
            Mapped::Ignore
        );
        assert_eq!(
            mapper.map(&AgentEvent::thinking("m2", "whole thought")),
            Mapped::Update(vec![json!({
                "sessionUpdate": "agent_thought_chunk",
                "content": { "type": "text", "text": "whole thought" },
            })])
        );
    }

    #[test]
    fn golden_tool_call_start_maps_to_pending_tool_call() {
        let mut mapper = EventMapper::default();
        let event = AgentEvent::tool_call_start(
            "tc-1",
            "grep",
            "Search files",
            json!({"pattern": "needle"}),
            ToolTier::Observe,
        );
        assert_eq!(
            mapper.map(&event),
            Mapped::Update(vec![json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "tc-1",
                "title": "Search files",
                "kind": "search",
                "status": "pending",
                "rawInput": {"pattern": "needle"},
            })])
        );
    }

    #[test]
    fn golden_tool_progress_maps_to_in_progress_update() {
        let mut mapper = EventMapper::default();
        let event = AgentEvent::tool_progress(
            "tc-1",
            "grep",
            "Search files",
            "scanning",
            "3 files checked",
            Some(json!({"files": 3})),
        );
        assert_eq!(
            mapper.map(&event),
            Mapped::Update(vec![tool_update("tc-1", "in_progress", "3 files checked")])
        );
    }

    #[test]
    fn golden_tool_call_end_maps_success_and_failure_with_output() {
        let mut mapper = EventMapper::default();
        for (result, status, output) in [
            (ToolResult::success("found it"), "completed", "found it"),
            (ToolResult::error("not found"), "failed", "not found"),
        ] {
            let event = AgentEvent::tool_call_end("tc-1", "grep", "Search files", result);
            assert_eq!(
                mapper.map(&event),
                Mapped::Update(vec![tool_update("tc-1", status, output)])
            );
        }
    }

    /// §3.1 rules 1, 2 and 4: the spawning `ToolCallStart` opens the row
    /// like any tool; ticks keep it `in_progress` with a summary; only
    /// `completed: true` closes it — as a directive carrying the
    /// invocation task id, so the run loop can fetch the result text.
    #[test]
    fn golden_subagent_lifecycle_opens_ticks_and_closes_only_on_completion() {
        let mut mapper = EventMapper::default();
        let start = AgentEvent::tool_call_start(
            "sub-1",
            "subagent_explore",
            "explore",
            json!({"task": "look around"}),
            ToolTier::Observe,
        );
        assert_eq!(
            mapper.map(&start),
            Mapped::Update(vec![json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "sub-1",
                "title": "explore",
                "kind": "other",
                "status": "pending",
                "rawInput": {"task": "look around"},
            })])
        );
        assert_eq!(
            mapper.map(&subagent_progress("sub-1", false, false, 2)),
            Mapped::Update(vec![tool_update(
                "sub-1",
                "in_progress",
                "Zara (explore) · turn 2/10 · 3 tools · 1234 tokens",
            )])
        );
        let closed = mapper.map(&subagent_progress("sub-1", true, true, 4));
        let Mapped::SubagentClosed(closed) = closed else {
            panic!("completion must be a SubagentClosed directive, got {closed:?}");
        };
        assert_eq!(closed.tool_call_id, "sub-1");
        assert_eq!(closed.subagent_task_id.as_deref(), Some("invocation-1"));
        assert!(closed.success);
        assert_eq!(
            closed.update(Some("The answer is 42.")),
            tool_update("sub-1", "completed", "The answer is 42.")
        );
        assert_eq!(
            closed.update(None),
            tool_update(
                "sub-1",
                "completed",
                "Zara (explore) · turn 4/10 · 3 tools · 1234 tokens",
            )
        );
    }

    #[test]
    fn golden_subagent_failure_closes_failed_and_retried_completion_is_ignored() {
        let mut mapper = EventMapper::default();
        let Mapped::SubagentClosed(closed) =
            mapper.map(&subagent_progress("sub-1", true, false, 1))
        else {
            panic!("completion must be a SubagentClosed directive");
        };
        assert!(!closed.success);
        assert_eq!(closed.update(None)["status"], json!("failed"));
        // A retried completion commit, or a tick that raced the close,
        // must not reopen or re-close the row.
        assert_eq!(
            mapper.map(&subagent_progress("sub-1", true, false, 1)),
            Mapped::Ignore
        );
        assert_eq!(
            mapper.map(&subagent_progress("sub-1", false, false, 1)),
            Mapped::Ignore
        );
    }

    #[test]
    fn subagent_summary_degrades_without_nickname_or_turn_bounds() {
        assert_eq!(
            subagent_summary("explore", None, Some(1), None, 1, 10),
            "explore · turn 1 · 1 tool · 10 tokens"
        );
        assert_eq!(
            subagent_summary("explore", None, None, Some(5), 0, 0),
            "explore · 0 tools · 0 tokens"
        );
    }

    /// Plan synthesis waits for the SUCCESSFUL `ToolCallEnd`: the tool
    /// row closes first, then the plan follows in the same mapping.
    #[test]
    fn golden_successful_todo_write_synthesizes_a_plan_after_its_tool_row() {
        let mut mapper = EventMapper::default();
        let input = json!({"todos": [
            {"content": "Read the ticket", "status": "completed", "activeForm": "Reading"},
            {"content": "Write the fix", "status": "in_progress", "activeForm": "Writing"},
            {"content": "Open the PR", "status": "pending", "activeForm": "Opening"},
        ]});
        let start = AgentEvent::tool_call_start(
            "todo-1",
            "todo_write",
            "Update Tasks",
            input,
            ToolTier::Observe,
        );
        assert!(matches!(mapper.map(&start), Mapped::Update(_)));
        let end = AgentEvent::tool_call_end(
            "todo-1",
            "todo_write",
            "Update Tasks",
            ToolResult::success("3 items"),
        );
        assert_eq!(
            mapper.map(&end),
            Mapped::Update(vec![
                tool_update("todo-1", "completed", "3 items"),
                json!({
                    "sessionUpdate": "plan",
                    "entries": [
                        {"content": "Read the ticket", "priority": "medium", "status": "completed"},
                        {"content": "Write the fix", "priority": "medium", "status": "in_progress"},
                        {"content": "Open the PR", "priority": "medium", "status": "pending"},
                    ],
                }),
            ])
        );
        // The cache is released: a second end for the same id is a plain
        // tool update.
        assert_eq!(
            mapper.map(&end),
            Mapped::Update(vec![tool_update("todo-1", "completed", "3 items")])
        );
    }

    #[test]
    fn failed_todo_write_surfaces_no_plan_and_releases_its_cache() {
        let mut mapper = EventMapper::default();
        let input = json!({"todos": [{"content": "x", "status": "pending", "activeForm": "y"}]});
        let start = AgentEvent::tool_call_start(
            "todo-1",
            "todo_write",
            "Update Tasks",
            input,
            ToolTier::Observe,
        );
        mapper.map(&start);
        let failed = AgentEvent::tool_call_end(
            "todo-1",
            "todo_write",
            "Update Tasks",
            ToolResult::error("Invalid input for todo_write"),
        );
        assert_eq!(
            mapper.map(&failed),
            Mapped::Update(vec![tool_update(
                "todo-1",
                "failed",
                "Invalid input for todo_write"
            )])
        );
        let later_success = AgentEvent::tool_call_end(
            "todo-1",
            "todo_write",
            "Update Tasks",
            ToolResult::success("ok"),
        );
        assert_eq!(
            mapper.map(&later_success),
            Mapped::Update(vec![tool_update("todo-1", "completed", "ok")]),
            "a released cache must not resurrect a plan from a stale input"
        );
    }

    #[test]
    fn golden_turn_complete_maps_usage_telemetry() {
        let mut mapper = EventMapper::default();
        let usage = TokenUsage {
            input_tokens: 11,
            output_tokens: 7,
            cached_input_tokens: 3,
            cache_creation_input_tokens: 2,
        };
        assert_eq!(
            mapper.map(&AgentEvent::turn_complete(2, usage)),
            Mapped::Update(vec![json!({
                "sessionUpdate": "usage_update",
                "turn": 2,
                "inputTokens": 11,
                "outputTokens": 7,
                "cachedInputTokens": 3,
                "cacheCreationInputTokens": 2,
            })])
        );
    }

    #[test]
    fn golden_terminal_variants_keep_their_stop_reasons() {
        let mut mapper = EventMapper::default();
        assert_eq!(mapper.map(&done()), Mapped::Terminal(StopReason::EndTurn));
        assert_eq!(
            mapper.map(&AgentEvent::cancelled(1, TokenUsage::default())),
            Mapped::Terminal(StopReason::Cancelled)
        );
        assert_eq!(
            mapper.map(&AgentEvent::refusal("m1", None)),
            Mapped::Terminal(StopReason::Refusal)
        );
        assert_eq!(
            mapper.map(&AgentEvent::budget_exceeded(
                ThreadId::from_string("t".to_owned()),
                1,
                TokenUsage::default(),
                Duration::from_millis(1),
                None,
                agent_sdk_foundation::types::BudgetLimitKind::TotalTokens,
            )),
            Mapped::Terminal(StopReason::MaxTokens)
        );
        assert_eq!(
            mapper.map(&AgentEvent::error("boom", false)),
            Mapped::Fail("boom".to_owned())
        );
    }

    #[test]
    fn drop_list_variants_are_absent() {
        let mut mapper = EventMapper::default();
        let events = [
            AgentEvent::AutoRetryStart {
                attempt: 1,
                max_attempts: 3,
                delay_ms: 10,
                error_message: "retry".to_owned(),
            },
            AgentEvent::AutoRetryEnd {
                attempt: 1,
                success: true,
                final_error: None,
            },
            AgentEvent::context_compacted(10, 4, 1_000, 400),
        ];
        for event in events {
            assert_eq!(mapper.map(&event), Mapped::Ignore);
        }
    }

    #[test]
    fn kind_table_covers_every_sdk_builtin_and_unknown_is_other() {
        let expected = [
            ("read", "read"),
            ("write", "execute"),
            ("edit", "execute"),
            ("multi_edit", "execute"),
            ("bash", "execute"),
            ("glob", "search"),
            ("grep", "search"),
            ("notebook_read", "read"),
            ("notebook_edit", "execute"),
            ("todo_read", "think"),
            ("todo_write", "think"),
            ("ask_user", "think"),
            ("link_fetch", "fetch"),
            ("web_search", "search"),
        ];
        for (name, kind) in expected {
            assert_eq!(tool_kind(name), kind, "wrong ACP kind for {name}");
        }
        assert_eq!(tool_kind("remote_mcp_tool"), "other");
        assert_eq!(tool_kind(""), "other");
    }

    /// Every tool name satoshi registers, with the coarse ACP kind this
    /// slice assigns it. Mutating and managing tools stay `other` on
    /// purpose; everything with a specific kind must classify to it.
    const SATOSHI_TOOL_KINDS: &[(&str, &str)] = &[
        ("advisor", "think"),
        ("analytics_describe_table", "read"),
        ("analytics_list_tables", "read"),
        ("analytics_query", "execute"),
        ("bash", "execute"),
        ("buzz_send_message", "other"),
        ("check_integration_health", "read"),
        ("conversation_recall", "search"),
        ("create_integration_skill", "other"),
        ("create_objective", "other"),
        ("create_scheduled_task", "other"),
        ("deep_research", "search"),
        ("delete_skill", "other"),
        ("disable_skill", "other"),
        ("enable_skill", "other"),
        ("execute_code", "execute"),
        ("fetch_page", "fetch"),
        ("fraud_artifact", "other"),
        ("fraud_complete_run", "other"),
        ("fraud_execute_code", "execute"),
        ("fraud_source_capture", "other"),
        ("fraud_source_export", "other"),
        ("fraud_source_query", "execute"),
        ("fraud_source_schema", "read"),
        ("get_document", "read"),
        ("get_employee_work", "read"),
        ("get_okr_status", "read"),
        ("get_rock_detail", "read"),
        ("get_skill_versions", "read"),
        ("get_team_alignment", "read"),
        ("graph_context_pack", "read"),
        ("graph_get", "read"),
        ("graph_neighbors", "read"),
        ("graph_search", "search"),
        ("investigation_edit", "other"),
        ("investigation_list", "read"),
        ("investigation_move", "other"),
        ("investigation_read", "read"),
        ("investigation_search", "search"),
        ("investigation_write", "other"),
        ("knowledge_forget", "other"),
        ("knowledge_recall", "search"),
        ("knowledge_remember", "other"),
        ("linear_create_cycle", "other"),
        ("linear_create_initiative", "other"),
        ("linear_create_issue", "other"),
        ("linear_create_project", "other"),
        ("linear_update_issue", "other"),
        ("list_scheduled_tasks", "read"),
        ("list_skills", "read"),
        ("load_skill", "read"),
        ("manage_scheduled_task", "other"),
        ("parse_document", "read"),
        ("pg_describe_table", "read"),
        ("pg_list_tables", "read"),
        ("pg_query", "execute"),
        ("read_logs", "read"),
        ("read_pdf", "read"),
        ("record_snapshot", "other"),
        ("rlm", "execute"),
        ("rollback_skill_version", "other"),
        ("sandbox_coding_agent", "execute"),
        ("scratch_read", "read"),
        ("scratch_write", "other"),
        ("search_knowledge", "search"),
        ("slack_send_message", "other"),
        ("store_integration_secret", "other"),
        ("submit_integration_approval", "other"),
        ("web_search", "search"),
    ];

    #[test]
    fn kind_table_covers_every_satoshi_registered_tool() {
        const VALID_KINDS: [&str; 6] = ["read", "search", "execute", "fetch", "think", "other"];
        for (name, kind) in SATOSHI_TOOL_KINDS {
            let mapped = tool_kind(name);
            assert_eq!(mapped, *kind, "wrong ACP kind for {name}");
            assert!(
                VALID_KINDS.contains(&mapped),
                "{name} mapped outside the coarse kind set: {mapped}"
            );
        }
    }
}
