//! `GenAI` payload conversion from SDK types to semconv JSON.
//!
//! Converts SDK [`ChatRequest`] / [`ChatResponse`] types into the JSON
//! schemas defined by the `GenAI` semantic conventions for
//! `gen_ai.system_instructions`, `gen_ai.input.messages`, and
//! `gen_ai.output.messages`.
//!
//! # Privacy
//!
//! This layer sits directly in front of observability egress —
//! whatever shape we emit here flows into `OTel` spans, `ObservabilityStore`
//! implementations, and ultimately into third-party collectors
//! (Langfuse, Datadog, honeycomb, …). [`PayloadRedactor`] wraps a
//! pluggable [`PiiDetector`] and masks PII in every text and JSON
//! string leaf before they leave this module, so the two conversion
//! paths (directly serialized onto spans, or handed to the store)
//! share one audited redaction pass.
//!
//! Callers that do not need redaction — including existing call sites
//! that have not adopted the detector yet — should use
//! [`PayloadRedactor::noop`] (or the module-level free functions,
//! which delegate to it). Financial and other PII-sensitive
//! integrations should construct a redactor with
//! [`agent_sdk_core::privacy::BaselineDetector`] or a custom
//! detector.

use crate::llm::{ChatRequest, ChatResponse, Content, ContentBlock, Message, Role};
use agent_sdk_core::privacy::{NoopDetector, PiiDetector, mask_spans};
use serde_json::{Value, json};
use std::sync::Arc;

use super::attrs::finish_reason_str;

/// Redacts PII from payloads before they flow to observability sinks.
///
/// Wraps a [`PiiDetector`] and applies it to every text and JSON
/// string leaf emitted by [`convert_system_instructions`](Self::convert_system_instructions),
/// [`convert_input_messages`](Self::convert_input_messages), and
/// [`convert_output_messages`](Self::convert_output_messages).
///
/// Construct with [`PayloadRedactor::new`] wrapping any
/// `Arc<dyn PiiDetector>`, or with [`PayloadRedactor::noop`] for the
/// pass-through case. The redactor is `Clone` and `Send + Sync`, so
/// one instance can be shared across the agent loop for the lifetime
/// of a run.
#[derive(Clone)]
pub struct PayloadRedactor {
    detector: Arc<dyn PiiDetector>,
}

impl PayloadRedactor {
    /// Wrap an existing detector.
    #[must_use]
    pub fn new(detector: Arc<dyn PiiDetector>) -> Self {
        Self { detector }
    }

    /// Redactor that performs no masking — produces byte-identical
    /// JSON output to the raw conversion path.
    #[must_use]
    pub fn noop() -> Self {
        Self {
            detector: Arc::new(NoopDetector),
        }
    }

    /// Convert system instructions, masking PII in the system prompt.
    ///
    /// Returns `None` if the system prompt is empty.
    #[must_use]
    pub fn convert_system_instructions(&self, request: &ChatRequest) -> Option<Value> {
        if request.system.is_empty() {
            return None;
        }
        Some(json!([{"text": self.mask_str(&request.system)}]))
    }

    /// Convert input messages into semconv JSON, masking PII in every
    /// text and tool-argument leaf.
    #[must_use]
    pub fn convert_input_messages(&self, request: &ChatRequest) -> Value {
        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(|m| self.convert_message(m))
            .collect();
        Value::Array(messages)
    }

    /// Convert a [`ChatResponse`] into semconv output-messages JSON,
    /// masking PII in every assistant text, thinking text, and
    /// tool-argument leaf.
    #[must_use]
    pub fn convert_output_messages(&self, response: &ChatResponse) -> Value {
        let parts: Vec<Value> = response
            .content
            .iter()
            .filter_map(|b| self.convert_block(b))
            .collect();
        let mut message = json!({
            "role": "assistant",
            "content": Value::Array(parts),
        });
        if let Some(reason) = response.stop_reason {
            message["finish_reason"] = json!(finish_reason_str(reason));
        }
        json!([message])
    }

    fn convert_message(&self, message: &Message) -> Value {
        let role = match message.role {
            Role::User => determine_user_message_role(message),
            Role::Assistant => "assistant",
        };
        let content = self.convert_content(&message.content);
        json!({
            "role": role,
            "content": content,
        })
    }

    fn convert_content(&self, content: &Content) -> Value {
        match content {
            Content::Text(text) => json!([{"text": self.mask_str(text)}]),
            Content::Blocks(blocks) => {
                let parts: Vec<Value> = blocks
                    .iter()
                    .filter_map(|b| self.convert_block(b))
                    .collect();
                Value::Array(parts)
            }
        }
    }

    fn convert_block(&self, block: &ContentBlock) -> Option<Value> {
        match block {
            ContentBlock::Text { text } => Some(json!({"text": self.mask_str(text)})),
            ContentBlock::Thinking { thinking, .. } => Some(json!({
                "type": "reasoning",
                "text": self.mask_str(thinking),
            })),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                let masked_input = self.mask_json(input);
                Some(json!({
                    "type": "tool_call",
                    "id": id,
                    "name": name,
                    "arguments": masked_input.to_string(),
                }))
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let mut part = json!({
                    "type": "tool_call_response",
                    "id": tool_use_id,
                    "output": self.mask_str(content),
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
            // Redacted-thinking blocks carry nothing to surface; this also
            // omits unknown future `#[non_exhaustive]` block kinds from the
            // observability payload.
            _ => None,
        }
    }

    /// Mask PII in a plain string.
    fn mask_str(&self, text: &str) -> String {
        let spans = self.detector.detect(text);
        if spans.is_empty() {
            text.to_owned()
        } else {
            mask_spans(text, &spans)
        }
    }

    /// Recursively mask every string leaf within a JSON value.
    fn mask_json(&self, value: &Value) -> Value {
        match value {
            Value::String(s) => Value::String(self.mask_str(s)),
            Value::Array(arr) => Value::Array(arr.iter().map(|v| self.mask_json(v)).collect()),
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), self.mask_json(v)))
                    .collect(),
            ),
            Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
        }
    }
}

impl Default for PayloadRedactor {
    fn default() -> Self {
        Self::noop()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Free function façade — delegate to `PayloadRedactor::noop`.
// Existing callers keep their call sites untouched; new integrations
// should construct a `PayloadRedactor` explicitly.
// ─────────────────────────────────────────────────────────────────────

/// Convert system instructions from a `ChatRequest` into semconv JSON.
///
/// Returns `None` if the system prompt is empty.
#[must_use]
pub fn convert_system_instructions(request: &ChatRequest) -> Option<Value> {
    PayloadRedactor::noop().convert_system_instructions(request)
}

/// Convert input messages from a `ChatRequest` into semconv JSON.
#[must_use]
pub fn convert_input_messages(request: &ChatRequest) -> Value {
    PayloadRedactor::noop().convert_input_messages(request)
}

/// Convert a `ChatResponse` into semconv output messages JSON.
///
/// Returns a JSON array with one assistant message per response
/// (the SDK currently returns a single candidate).
#[must_use]
pub fn convert_output_messages(response: &ChatResponse) -> Value {
    PayloadRedactor::noop().convert_output_messages(response)
}

/// Decide whether a User-role message is actually a `tool` message
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatRequest, ChatResponse, ContentSource, StopReason, Usage};
    use agent_sdk_core::privacy::BaselineDetector;

    fn empty_request(system: &str, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            system: system.to_owned(),
            messages,
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: false,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
        }
    }

    // ── Noop redactor / free functions preserve existing behaviour ──

    #[test]
    fn empty_system_returns_none() {
        let request = empty_request("", vec![]);
        assert!(convert_system_instructions(&request).is_none());
    }

    #[test]
    fn system_instructions_wraps_in_text_array() {
        let request = empty_request("You are helpful.", vec![]);
        let result = convert_system_instructions(&request).expect("should be Some");
        assert_eq!(result, json!([{"text": "You are helpful."}]));
    }

    #[test]
    fn user_text_message_converts_correctly() {
        let msg = Message::user("Hello");
        let result = PayloadRedactor::noop().convert_message(&msg);
        assert_eq!(result["role"], "user");
        assert_eq!(result["content"][0]["text"], "Hello");
    }

    #[test]
    fn assistant_text_message_converts_correctly() {
        let msg = Message::assistant("Hi there");
        let result = PayloadRedactor::noop().convert_message(&msg);
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
        let result = PayloadRedactor::noop().convert_message(&msg);
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
        let result = PayloadRedactor::noop().convert_message(&msg);
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
        let result = PayloadRedactor::noop().convert_message(&msg);
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
        let result = PayloadRedactor::noop().convert_message(&msg);
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
        let result = PayloadRedactor::noop().convert_message(&msg);
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
        let result = PayloadRedactor::noop().convert_message(&msg);
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
                cache_creation_input_tokens: 0,
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
                cache_creation_input_tokens: 0,
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
        let result = PayloadRedactor::noop().convert_message(&msg);
        assert_eq!(result["content"][0]["is_error"], true);
    }

    #[test]
    fn input_messages_preserves_order() {
        let request = empty_request(
            "",
            vec![
                Message::user("first"),
                Message::assistant("second"),
                Message::user("third"),
            ],
        );
        let result = convert_input_messages(&request);
        let arr = result.as_array().expect("array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["role"], "user");
        assert_eq!(arr[1]["role"], "assistant");
        assert_eq!(arr[2]["role"], "user");
    }

    // ── Baseline redactor masks PII across every covered surface ──

    fn baseline_redactor() -> PayloadRedactor {
        PayloadRedactor::new(Arc::new(
            BaselineDetector::new().expect("baseline compiles"),
        ))
    }

    #[test]
    fn redacts_email_in_system_prompt() {
        let request = empty_request("Contact support at ops@example.com.", vec![]);
        let result = baseline_redactor()
            .convert_system_instructions(&request)
            .expect("some");
        let text = result[0]["text"].as_str().expect("text string");
        assert!(
            text.contains("[REDACTED:email]"),
            "system prompt not redacted: {text}"
        );
        assert!(!text.contains("ops@example.com"));
    }

    #[test]
    fn redacts_cpf_in_user_text() {
        let request = empty_request("", vec![Message::user("meu CPF é 111.444.777-35")]);
        let result = baseline_redactor().convert_input_messages(&request);
        let text = result[0]["content"][0]["text"].as_str().expect("text");
        assert!(text.contains("[REDACTED:cpf]"), "user text: {text}");
        assert!(!text.contains("111.444.777-35"));
    }

    #[test]
    fn redacts_pan_in_tool_result_output() {
        let msg = Message {
            role: Role::User,
            content: Content::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "c1".to_string(),
                content: "charged card 4111 1111 1111 1111 successfully".to_string(),
                is_error: None,
            }]),
        };
        let result = baseline_redactor().convert_message(&msg);
        let output = result["content"][0]["output"].as_str().expect("output");
        assert!(
            output.contains("[REDACTED:credit_card]"),
            "tool output: {output}"
        );
        assert!(!output.contains("4111 1111 1111 1111"));
    }

    #[test]
    fn redacts_strings_inside_tool_call_arguments_json() {
        let msg = Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![ContentBlock::ToolUse {
                id: "c1".to_string(),
                name: "send_pix".to_string(),
                input: json!({
                    "chave_pix": "ana@example.com",
                    "amount_brl": 100,
                    "metadata": {
                        "recipient_cpf": "111.444.777-35",
                        "note": "salário"
                    }
                }),
                thought_signature: None,
            }]),
        };
        let result = baseline_redactor().convert_message(&msg);
        let args = result["content"][0]["arguments"].as_str().expect("args");
        assert!(args.contains("[REDACTED:email]"), "args: {args}");
        assert!(args.contains("[REDACTED:cpf]"), "args: {args}");
        assert!(!args.contains("ana@example.com"));
        assert!(!args.contains("111.444.777-35"));
        // Non-string leaves are preserved as-is.
        assert!(args.contains("100"));
    }

    #[test]
    fn redacts_secret_in_assistant_output() {
        let response = ChatResponse {
            id: "r1".to_string(),
            content: vec![ContentBlock::Text {
                text: "here is the key sk-abcdefghijklmnopqrstuv for ci".to_string(),
            }],
            model: "m".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        };
        let result = baseline_redactor().convert_output_messages(&response);
        let text = result[0]["content"][0]["text"].as_str().expect("text");
        assert!(text.contains("[REDACTED:secret]"), "output: {text}");
        assert!(!text.contains("sk-abcdefghijklmnopqrstuv"));
    }

    #[test]
    fn redacts_pii_in_thinking_text() {
        let msg = Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![ContentBlock::Thinking {
                thinking: "User CPF is 111.444.777-35 — I should confirm before sending."
                    .to_string(),
                signature: None,
            }]),
        };
        let result = baseline_redactor().convert_message(&msg);
        let text = result["content"][0]["text"].as_str().expect("text");
        assert!(text.contains("[REDACTED:cpf]"), "thinking: {text}");
    }

    #[test]
    fn mask_json_preserves_non_string_leaves() {
        let input = json!({
            "amount": 42.5,
            "active": true,
            "items": null,
            "email": "ana@example.com"
        });
        let redacted = baseline_redactor().mask_json(&input);
        assert_eq!(redacted["amount"], json!(42.5));
        assert_eq!(redacted["active"], json!(true));
        assert_eq!(redacted["items"], json!(null));
        assert!(
            redacted["email"]
                .as_str()
                .expect("email")
                .contains("[REDACTED:email]")
        );
    }

    #[test]
    fn noop_redactor_produces_same_output_as_free_functions() {
        // Spot-check: a payload with no PII should serialize identically
        // via the noop redactor and the free functions.
        let request = empty_request("System text", vec![Message::user("Hello, world")]);
        assert_eq!(
            PayloadRedactor::noop().convert_input_messages(&request),
            convert_input_messages(&request),
        );
        assert_eq!(
            PayloadRedactor::noop().convert_system_instructions(&request),
            convert_system_instructions(&request),
        );
    }
}
