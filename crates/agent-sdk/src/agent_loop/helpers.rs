use super::types::ExtractedContent;
use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::hooks::AgentHooks;
use crate::llm::{ChatResponse, Content, ContentBlock, Message, Role};
use crate::stores::EventStore;
use crate::types::{AgentError, PendingToolCallInfo, RetryConfig, ThreadId};
use std::sync::Arc;
use std::time::Duration;

/// Saturating conversion from usize to u32.
pub(super) fn turns_to_u32(turns: usize) -> u32 {
    u32::try_from(turns).unwrap_or(u32::MAX)
}

/// Convert u128 milliseconds to u64, capping at `u64::MAX`
pub(super) fn millis_to_u64(millis: u128) -> u64 {
    u64::try_from(millis).unwrap_or(u64::MAX)
}

/// Calculate exponential backoff delay with jitter.
///
/// Uses exponential backoff with the formula: `base * 2^(attempt-1) + jitter`,
/// capped at the maximum delay. Jitter (0-1000ms) helps avoid thundering herd.
pub(super) fn calculate_backoff_delay(attempt: u32, config: &RetryConfig) -> Duration {
    // Exponential backoff: base, base*2, base*4, base*8, ...
    let base_delay = config
        .base_delay_ms
        .saturating_mul(1u64 << (attempt.saturating_sub(1)));

    // Add jitter (0-1000ms or 10% of base, whichever is smaller) to avoid thundering herd.
    // Uses RandomState for per-process randomness instead of system clock.
    let max_jitter = config.base_delay_ms.min(1000);
    let jitter = if max_jitter > 0 {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let h = RandomState::new().build_hasher().finish();
        h % max_jitter
    } else {
        0
    };

    let delay_ms = base_delay.saturating_add(jitter).min(config.max_delay_ms);
    Duration::from_millis(delay_ms)
}

pub(super) fn pending_tool_index(
    pending_tool_calls: &[PendingToolCallInfo],
    tool_id: &str,
) -> Result<usize, AgentError> {
    pending_tool_calls
        .iter()
        .position(|p| p.id == tool_id)
        .ok_or_else(|| AgentError::new(format!("Pending tool ID not found: {tool_id}"), false))
}

/// Extract content from an LLM response.
pub(super) fn extract_content(response: &ChatResponse) -> ExtractedContent {
    let mut thinking_parts = Vec::new();
    let mut text_parts = Vec::new();
    let mut tool_uses = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            ContentBlock::Thinking { thinking, .. } => {
                thinking_parts.push(thinking.clone());
            }
            ContentBlock::RedactedThinking { .. }
            | ContentBlock::ToolResult { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::Document { .. } => {
                // These blocks don't produce extractable content
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                let input = if input.is_null() {
                    serde_json::json!({})
                } else {
                    input.clone()
                };
                tool_uses.push((id.clone(), name.clone(), input.clone()));
            }
        }
    }

    let thinking = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join("\n"))
    };

    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    };

    (thinking, text, tool_uses)
}

/// Send an event to the authoritative turn observer.
pub(super) async fn send_event<H>(
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    turn: usize,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
    event: AgentEvent,
) -> Result<(), AgentError>
where
    H: AgentHooks,
{
    hooks.on_event(&event).await;

    let envelope = AgentEventEnvelope::wrap(event, seq);
    event_store
        .append(thread_id, turn, envelope)
        .await
        .map_err(|error| AgentError::new(format!("Failed to append event: {error}"), false))
}

/// Send an event directly to the store without going through hooks.
///
/// Used by async tool execution for progress events that bypass the hook system.
pub(super) async fn wrap_and_send(
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    turn: usize,
    event: AgentEvent,
    seq: &SequenceCounter,
) -> Result<(), AgentError> {
    let envelope = AgentEventEnvelope::wrap(event, seq);
    event_store
        .append(thread_id, turn, envelope)
        .await
        .map_err(|error| AgentError::new(format!("Failed to append event: {error}"), false))
}

pub(super) fn build_assistant_message(response: &ChatResponse) -> Message {
    let mut blocks = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => {
                blocks.push(ContentBlock::Text { text: text.clone() });
            }
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                blocks.push(ContentBlock::Thinking {
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                });
            }
            ContentBlock::RedactedThinking { data } => {
                blocks.push(ContentBlock::RedactedThinking { data: data.clone() });
            }
            ContentBlock::ToolResult { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::Document { .. } => {
                // These blocks shouldn't appear in assistant responses
            }
            ContentBlock::ToolUse {
                id,
                name,
                input,
                thought_signature,
            } => {
                blocks.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    thought_signature: thought_signature.clone(),
                });
            }
        }
    }

    Message {
        role: Role::Assistant,
        content: Content::Blocks(blocks),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatResponse, ContentBlock, Usage};
    use serde_json::json;

    #[test]
    fn test_extract_content_text_only() {
        let response = ChatResponse {
            id: "msg_1".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            model: "test".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
            },
        };

        let (thinking, text, tool_uses) = extract_content(&response);
        assert!(thinking.is_none());
        assert_eq!(text, Some("Hello".to_string()));
        assert!(tool_uses.is_empty());
    }

    #[test]
    fn test_extract_content_tool_use() {
        let response = ChatResponse {
            id: "msg_1".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "test_tool".to_string(),
                input: json!({"key": "value"}),
                thought_signature: None,
            }],
            model: "test".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
            },
        };

        let (thinking, text, tool_uses) = extract_content(&response);
        assert!(thinking.is_none());
        assert!(text.is_none());
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].1, "test_tool");
    }

    #[test]
    fn test_extract_content_mixed() {
        let response = ChatResponse {
            id: "msg_1".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Let me help".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "helper".to_string(),
                    input: json!({}),
                    thought_signature: None,
                },
            ],
            model: "test".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
            },
        };

        let (thinking, text, tool_uses) = extract_content(&response);
        assert!(thinking.is_none());
        assert_eq!(text, Some("Let me help".to_string()));
        assert_eq!(tool_uses.len(), 1);
    }

    #[test]
    fn test_millis_to_u64() {
        assert_eq!(millis_to_u64(0), 0);
        assert_eq!(millis_to_u64(1000), 1000);
        assert_eq!(millis_to_u64(u128::from(u64::MAX)), u64::MAX);
        assert_eq!(millis_to_u64(u128::from(u64::MAX) + 1), u64::MAX);
    }

    #[test]
    fn test_build_assistant_message() {
        let response = ChatResponse {
            id: "msg_1".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Response text".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "echo".to_string(),
                    input: json!({"message": "test"}),
                    thought_signature: None,
                },
            ],
            model: "test".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
            },
        };

        let msg = build_assistant_message(&response);
        assert_eq!(msg.role, Role::Assistant);

        if let Content::Blocks(blocks) = msg.content {
            assert_eq!(blocks.len(), 2);
        } else {
            panic!("Expected Content::Blocks");
        }
    }
}
