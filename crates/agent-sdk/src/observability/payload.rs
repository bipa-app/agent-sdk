//! `GenAI` payload conversion from SDK types to semconv JSON.
//!
//! Converts SDK `ChatRequest` / `ChatResponse` types into the JSON schemas
//! defined by the `GenAI` semantic conventions for `gen_ai.system_instructions`,
//! `gen_ai.input.messages`, and `gen_ai.output.messages`.

use crate::llm::{ChatRequest, ChatResponse, Content, ContentBlock, Message, Role};
use serde_json::{Value, json};

use super::attrs::finish_reason_str;

/// Convert system instructions from a `ChatRequest` into semconv JSON.
///
/// Returns `None` if the system prompt is empty.
#[must_use]
pub fn convert_system_instructions(request: &ChatRequest) -> Option<Value> {
    if request.system.is_empty() {
        return None;
    }
    Some(json!([{"text": request.system}]))
}

/// Convert input messages from a `ChatRequest` into semconv JSON.
#[must_use]
pub fn convert_input_messages(request: &ChatRequest) -> Value {
    let messages: Vec<Value> = request.messages.iter().map(convert_message).collect();
    Value::Array(messages)
}

/// Convert a `ChatResponse` into semconv output messages JSON.
///
/// Returns a JSON array with one assistant message per response
/// (the SDK currently returns a single candidate).
#[must_use]
pub fn convert_output_messages(response: &ChatResponse) -> Value {
    let parts: Vec<Value> = response.content.iter().filter_map(convert_block).collect();
    let content = Value::Array(parts);

    let mut message = json!({
        "role": "assistant",
        "content": content,
    });

    if let Some(reason) = response.stop_reason {
        message["finish_reason"] = json!(finish_reason_str(reason));
    }

    json!([message])
}

fn convert_message(message: &Message) -> Value {
    let role = match message.role {
        Role::User => determine_user_message_role(message),
        Role::Assistant => "assistant",
    };

    let content = convert_content(&message.content);

    json!({
        "role": role,
        "content": content,
    })
}

/// Determine whether a User-role message is actually a `tool` message
/// (SDK batches tool results as User messages).
fn determine_user_message_role(message: &Message) -> &'static str {
    match &message.content {
        Content::Blocks(blocks) => {
            let has_tool_result = blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
            if has_tool_result { "tool" } else { "user" }
        }
        Content::Text(_) => "user",
    }
}

fn convert_content(content: &Content) -> Value {
    match content {
        Content::Text(text) => json!([{"text": text}]),
        Content::Blocks(blocks) => {
            let parts: Vec<Value> = blocks.iter().filter_map(convert_block).collect();
            Value::Array(parts)
        }
    }
}

fn convert_block(block: &ContentBlock) -> Option<Value> {
    match block {
        ContentBlock::Text { text } => Some(json!({"text": text})),
        ContentBlock::Thinking { thinking, .. } => {
            Some(json!({"type": "reasoning", "text": thinking}))
        }
        ContentBlock::RedactedThinking { .. } => None,
        ContentBlock::ToolUse {
            id, name, input, ..
        } => Some(json!({
            "type": "tool_call",
            "id": id,
            "name": name,
            "arguments": input.to_string(),
        })),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let mut part = json!({
                "type": "tool_call_response",
                "id": tool_use_id,
                "output": content,
            });
            if *is_error == Some(true) {
                part["is_error"] = json!(true);
            }
            Some(part)
        }
        ContentBlock::Image { source } => Some(json!({
            "type": "blob",
            "mime_type": source.media_type,
            "modality": "image",
            "size": source.data.len(),
        })),
        ContentBlock::Document { source } => {
            let mut part = json!({
                "type": "blob",
                "mime_type": source.media_type,
                "size": source.data.len(),
            });
            if source.media_type.starts_with("image/") {
                part["modality"] = json!("image");
            }
            Some(part)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatRequest, ChatResponse, ContentSource, StopReason, Usage};

    #[test]
    fn empty_system_returns_none() {
        let request = ChatRequest {
            system: String::new(),
            messages: vec![],
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: false,
            session_id: None,
            cached_content: None,
            thinking: None,
        };
        assert!(convert_system_instructions(&request).is_none());
    }

    #[test]
    fn system_instructions_wraps_in_text_array() {
        let request = ChatRequest {
            system: "You are helpful.".to_string(),
            messages: vec![],
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: false,
            session_id: None,
            cached_content: None,
            thinking: None,
        };
        let result = convert_system_instructions(&request).expect("should be Some");
        assert_eq!(result, json!([{"text": "You are helpful."}]));
    }

    #[test]
    fn user_text_message_converts_correctly() {
        let msg = Message::user("Hello");
        let result = convert_message(&msg);
        assert_eq!(result["role"], "user");
        assert_eq!(result["content"][0]["text"], "Hello");
    }

    #[test]
    fn assistant_text_message_converts_correctly() {
        let msg = Message::assistant("Hi there");
        let result = convert_message(&msg);
        assert_eq!(result["role"], "assistant");
        assert_eq!(result["content"][0]["text"], "Hi there");
    }

    #[test]
    fn tool_result_batch_maps_to_tool_role() {
        let msg = Message {
            role: Role::User,
            content: Content::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "result data".to_string(),
                is_error: None,
            }]),
        };
        let result = convert_message(&msg);
        assert_eq!(result["role"], "tool");
        assert_eq!(result["content"][0]["type"], "tool_call_response");
        assert_eq!(result["content"][0]["id"], "call_1");
        assert_eq!(result["content"][0]["output"], "result data");
    }

    #[test]
    fn tool_result_with_image_attachment_stays_in_tool_message() {
        let msg = Message {
            role: Role::User,
            content: Content::Blocks(vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: "screenshot taken".to_string(),
                    is_error: None,
                },
                ContentBlock::Image {
                    source: ContentSource::new("image/png", "aWdv"),
                },
            ]),
        };
        let result = convert_message(&msg);
        assert_eq!(result["role"], "tool");
        assert_eq!(result["content"][0]["type"], "tool_call_response");
        assert_eq!(result["content"][1]["type"], "blob");
        assert_eq!(result["content"][1]["modality"], "image");
    }

    #[test]
    fn thinking_block_maps_to_reasoning_part() {
        let msg = Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "Let me think...".to_string(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "The answer is 42".to_string(),
                },
            ]),
        };
        let result = convert_message(&msg);
        assert_eq!(result["content"][0]["type"], "reasoning");
        assert_eq!(result["content"][0]["text"], "Let me think...");
        assert_eq!(result["content"][1]["text"], "The answer is 42");
    }

    #[test]
    fn redacted_thinking_is_omitted() {
        let msg = Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![
                ContentBlock::RedactedThinking {
                    data: "secret".to_string(),
                },
                ContentBlock::Text {
                    text: "visible".to_string(),
                },
            ]),
        };
        let result = convert_message(&msg);
        let content = result["content"].as_array().expect("array");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], "visible");
    }

    #[test]
    fn tool_use_block_maps_to_tool_call_part() {
        let msg = Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read".to_string(),
                input: json!({"path": "/tmp/test.rs"}),
                thought_signature: None,
            }]),
        };
        let result = convert_message(&msg);
        assert_eq!(result["content"][0]["type"], "tool_call");
        assert_eq!(result["content"][0]["id"], "call_1");
        assert_eq!(result["content"][0]["name"], "read");
    }

    #[test]
    fn document_block_maps_to_blob_part() {
        let msg = Message {
            role: Role::User,
            content: Content::Blocks(vec![ContentBlock::Document {
                source: ContentSource::new("application/pdf", "cGRm"),
            }]),
        };
        let result = convert_message(&msg);
        assert_eq!(result["content"][0]["type"], "blob");
        assert_eq!(result["content"][0]["mime_type"], "application/pdf");
        assert_eq!(result["content"][0]["size"], 4);
    }

    #[test]
    fn output_messages_includes_finish_reason() {
        let response = ChatResponse {
            id: "resp_1".to_string(),
            content: vec![ContentBlock::Text {
                text: "Done".to_string(),
            }],
            model: "test-model".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
            },
        };
        let result = convert_output_messages(&response);
        let msg = &result[0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["finish_reason"], "stop");
        assert_eq!(msg["content"][0]["text"], "Done");
    }

    #[test]
    fn output_messages_tool_call_finish_reason() {
        let response = ChatResponse {
            id: "resp_1".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "c1".to_string(),
                name: "bash".to_string(),
                input: json!({"command": "ls"}),
                thought_signature: None,
            }],
            model: "test-model".to_string(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
            },
        };
        let result = convert_output_messages(&response);
        assert_eq!(result[0]["finish_reason"], "tool_call");
    }

    #[test]
    fn tool_result_error_flag_is_preserved() {
        let msg = Message {
            role: Role::User,
            content: Content::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "failed".to_string(),
                is_error: Some(true),
            }]),
        };
        let result = convert_message(&msg);
        assert_eq!(result["content"][0]["is_error"], true);
    }

    #[test]
    fn input_messages_preserves_order() {
        let request = ChatRequest {
            system: String::new(),
            messages: vec![
                Message::user("first"),
                Message::assistant("second"),
                Message::user("third"),
            ],
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: false,
            session_id: None,
            cached_content: None,
            thinking: None,
        };
        let result = convert_input_messages(&request);
        let arr = result.as_array().expect("array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["role"], "user");
        assert_eq!(arr[1]["role"], "assistant");
        assert_eq!(arr[2]["role"], "user");
    }
}
