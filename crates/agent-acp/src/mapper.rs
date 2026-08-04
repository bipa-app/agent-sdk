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

/// SDK built-ins covered by the ACP integration. Keeping this as data makes
/// additions easy to audit and guarantees unknown or remotely supplied tool
/// names remain safe.
const TOOL_KIND_TABLE: &[(&str, &str)] = &[
    ("read", "read"),
    ("notebook_read", "read"),
    ("glob", "search"),
    ("grep", "search"),
    ("web_search", "search"),
    ("bash", "execute"),
    ("write", "execute"),
    ("edit", "execute"),
    ("multi_edit", "execute"),
    ("notebook_edit", "execute"),
    ("link_fetch", "fetch"),
    ("todo_read", "think"),
    ("todo_write", "think"),
    ("ask_user", "think"),
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
}
