//! Schema-validated structured output.
//!
//! Demonstrates [`run_structured`]: instead of free-form text, the model's
//! final answer is constrained to a JSON Schema, validated, and bounded-
//! re-prompted on mismatch before failing with a typed error. The returned
//! [`StructuredOutput::value`] is guaranteed to satisfy the schema.
//!
//! This example uses a tiny stub provider that emits a forced `respond`
//! tool-call (the Anthropic-style tool-forcing path), so it compiles **and
//! runs** offline with no API key. Swap the stub for `AnthropicProvider` /
//! `OpenAiProvider` for real model output — the call site is identical.
//!
//! # Running
//!
//! ```bash
//! cargo run --example structured_output
//! ```

use agent_sdk::llm::{
    ChatOutcome, ChatRequest, ChatResponse, ContentBlock, LlmProvider, Message, ResponseFormat,
    StopReason, Usage,
};
use agent_sdk::{StructuredConfig, run_structured};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

/// The shape we want the model to return.
#[derive(Debug, Deserialize)]
struct Person {
    name: String,
    age: u32,
}

/// A stub provider that replies with a forced `respond` tool-call carrying a
/// schema-valid object. Reports `ToolForcing` support (the default), so the
/// structured runner reads the value out of the tool call.
struct StubProvider {
    model: String,
}

impl StubProvider {
    fn new() -> Self {
        Self {
            model: "stub-model".to_string(),
        }
    }
}

#[async_trait]
impl LlmProvider for StubProvider {
    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatOutcome> {
        // The runner injects a single forced tool; echo a valid value back.
        let tool_name = request
            .tools
            .as_ref()
            .and_then(|t| t.first())
            .map_or("respond", |t| t.name.as_str())
            .to_string();
        Ok(ChatOutcome::Success(ChatResponse {
            id: "stub".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: tool_name,
                input: json!({ "name": "Ada Lovelace", "age": 36 }),
                thought_signature: None,
            }],
            model: self.model().to_string(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }))
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn provider(&self) -> &'static str {
        "stub"
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer", "minimum": 0 }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });

    // `ChatRequest::new` keeps the call site tight — set only what differs from
    // the defaults (here: the response format).
    let request = ChatRequest::new(
        "Extract the person described by the user.",
        vec![Message::user("Ada Lovelace is 36.")],
    )
    .with_response_format(ResponseFormat::new("person", schema));

    let out = run_structured(&StubProvider::new(), request, StructuredConfig::default()).await?;

    // `out.value` is guaranteed schema-valid; deserialize into the Rust type.
    let person: Person = serde_json::from_value(out.value.clone())?;
    println!("validated value : {}", out.value);
    println!("re-prompts       : {}", out.retries);
    println!("typed            : {} (age {})", person.name, person.age);

    Ok(())
}
