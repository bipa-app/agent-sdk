//! Maps durable SDK events into ACP `session/update` payloads.

use std::collections::HashSet;

use agent_sdk_foundation::{AgentEvent, TokenUsage};
use serde_json::{Value, json};

use crate::wire::StopReason;

/// Outcome of mapping one committed event.
#[derive(Debug, PartialEq, Eq)]
pub enum Mapped {
    /// Emit a `session/update` payload.
    Update(Value),
    /// The turn is over with this stop reason.
    Terminal(StopReason),
    /// The turn failed; resolve the prompt as a JSON-RPC error.
    Fail(String),
    /// This event has no ACP representation in this slice.
    Ignore,
}

/// Per-prompt state needed to suppress consolidated content after deltas.
#[derive(Debug, Default)]
pub struct EventMapper {
    text_delta_seen: HashSet<String>,
    thinking_delta_seen: HashSet<String>,
}

impl EventMapper {
    pub fn map(&mut self, event: &AgentEvent) -> Mapped {
        match event {
            AgentEvent::TextDelta {
                message_id, delta, ..
            } => {
                self.text_delta_seen.insert(message_id.clone());
                Mapped::Update(text_chunk("agent_message_chunk", delta))
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
                Mapped::Update(text_chunk("agent_thought_chunk", delta))
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
            } => Mapped::Update(tool_call_start(id, name, display_name, input)),
            AgentEvent::ToolProgress { id, message, .. } => {
                Mapped::Update(tool_progress(id, message))
            }
            AgentEvent::ToolCallEnd { id, result, .. } => {
                Mapped::Update(tool_call_end(id, result.success, &result.output))
            }
            AgentEvent::TurnComplete { turn, usage, .. } => {
                Mapped::Update(usage_update(*turn, usage))
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

fn map_consolidated(
    delta_seen: &HashSet<String>,
    message_id: &str,
    text: &str,
    update_kind: &str,
) -> Mapped {
    if delta_seen.contains(message_id) {
        Mapped::Ignore
    } else {
        Mapped::Update(text_chunk(update_kind, text))
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

    #[test]
    fn golden_text_delta_and_consolidated_dedupe() {
        let mut mapper = EventMapper::default();
        assert_eq!(
            mapper.map(&AgentEvent::text_delta("m1", "hel")),
            Mapped::Update(json!({
                "sessionUpdate": "agent_message_chunk",
                "content": { "type": "text", "text": "hel" },
            }))
        );
        assert_eq!(mapper.map(&AgentEvent::text("m1", "hello")), Mapped::Ignore);
        assert_eq!(
            mapper.map(&AgentEvent::text("m2", "whole")),
            Mapped::Update(json!({
                "sessionUpdate": "agent_message_chunk",
                "content": { "type": "text", "text": "whole" },
            }))
        );
    }

    #[test]
    fn golden_thinking_delta_and_consolidated_dedupe() {
        let mut mapper = EventMapper::default();
        assert_eq!(
            mapper.map(&AgentEvent::thinking_delta("m1", "hmm")),
            Mapped::Update(json!({
                "sessionUpdate": "agent_thought_chunk",
                "content": { "type": "text", "text": "hmm" },
            }))
        );
        assert_eq!(
            mapper.map(&AgentEvent::thinking("m1", "hmm, yes")),
            Mapped::Ignore
        );
        assert_eq!(
            mapper.map(&AgentEvent::thinking("m2", "whole thought")),
            Mapped::Update(json!({
                "sessionUpdate": "agent_thought_chunk",
                "content": { "type": "text", "text": "whole thought" },
            }))
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
            Mapped::Update(json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "tc-1",
                "title": "Search files",
                "kind": "search",
                "status": "pending",
                "rawInput": {"pattern": "needle"},
            }))
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
            Mapped::Update(json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc-1",
                "status": "in_progress",
                "content": [{
                    "type": "content",
                    "content": { "type": "text", "text": "3 files checked" },
                }],
            }))
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
                Mapped::Update(json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-1",
                    "status": status,
                    "content": [{
                        "type": "content",
                        "content": { "type": "text", "text": output },
                    }],
                }))
            );
        }
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
            Mapped::Update(json!({
                "sessionUpdate": "usage_update",
                "turn": 2,
                "inputTokens": 11,
                "outputTokens": 7,
                "cachedInputTokens": 3,
                "cacheCreationInputTokens": 2,
            }))
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
