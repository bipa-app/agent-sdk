use super::types::ExtractedContent;
use crate::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
use crate::hooks::AgentHooks;
use crate::llm::{ChatResponse, Content, ContentBlock, Message, Role};
use crate::types::{AgentError, PendingToolCallInfo, RetryConfig};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// Saturating conversion from usize to u32.
#[allow(clippy::cast_possible_truncation)]
pub(super) const fn turns_to_u32(turns: usize) -> u32 {
    if turns > u32::MAX as usize {
        u32::MAX
    } else {
        turns as u32
    }
}

/// Convert u128 milliseconds to u64, capping at `u64::MAX`
#[allow(clippy::cast_possible_truncation)]
pub(super) const fn millis_to_u64(millis: u128) -> u64 {
    if millis > u64::MAX as u128 {
        u64::MAX
    } else {
        millis as u64
    }
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

    // Add jitter (0-1000ms or 10% of base, whichever is smaller) to avoid thundering herd
    let max_jitter = config.base_delay_ms.min(1000);
    let jitter = if max_jitter > 0 {
        u64::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos(),
        ) % max_jitter
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
            ContentBlock::RedactedThinking { .. } | ContentBlock::ToolResult { .. } => {
                // Redacted thinking is opaque; ToolResult shouldn't appear in response
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

/// Send an event to the consumer channel with non-blocking behavior.
///
/// This function first calls the hook's `on_event` method, then attempts to send
/// the event to the consumer channel. The sending behavior is designed to be
/// resilient to slow or disconnected consumers:
///
/// 1. First attempts a non-blocking send via `try_send`
/// 2. If the channel is full, waits up to 30 seconds for space
/// 3. If the channel is closed, logs and continues without blocking
/// 4. On timeout, logs an error and continues
///
/// This ensures that the agent loop doesn't block indefinitely if the consumer
/// is slow or has disconnected.
pub(super) async fn send_event<H>(
    tx: &mpsc::Sender<AgentEventEnvelope>,
    hooks: &Arc<H>,
    seq: &SequenceCounter,
    event: AgentEvent,
) where
    H: AgentHooks,
{
    hooks.on_event(&event).await;

    let envelope = AgentEventEnvelope::wrap(event, seq);

    // Try non-blocking send first to detect backpressure
    match tx.try_send(envelope) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(envelope)) => {
            // Channel is full - consumer is slow or blocked
            log::debug!("Event channel full, waiting for consumer...");
            // Fall back to blocking send with timeout
            match tokio::time::timeout(std::time::Duration::from_secs(30), tx.send(envelope)).await
            {
                Ok(Ok(())) => {}
                Ok(Err(_)) => {
                    log::warn!("Event channel closed while sending - consumer disconnected");
                }
                Err(_) => {
                    log::error!("Timeout waiting to send event - consumer may be deadlocked");
                }
            }
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            log::debug!("Event channel closed - consumer disconnected");
        }
    }
}

/// Send an event directly to the channel without going through hooks.
///
/// Used by async tool execution for progress events that bypass the hook system.
pub(super) async fn wrap_and_send(
    tx: &mpsc::Sender<AgentEventEnvelope>,
    event: AgentEvent,
    seq: &SequenceCounter,
) {
    let envelope = AgentEventEnvelope::wrap(event, seq);
    let _ = tx.send(envelope).await;
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
            ContentBlock::ToolResult { .. } => {
                // ToolResult shouldn't appear in response, but ignore if it does
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
