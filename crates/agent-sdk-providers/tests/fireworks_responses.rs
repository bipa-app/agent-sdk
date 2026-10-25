//! Fireworks on the Responses API with `store=false`.
//!
//! Tier B: a local [`wiremock`] server replays the response shapes Fireworks
//! documents (`tests/cassettes/fireworks_responses_*`), and every request the
//! provider sends is captured so the wire body can be asserted. No network,
//! no keys. The cassettes are built from the documented shapes; record live
//! ones with the Tier C test below to replace them.

#![cfg(feature = "openai")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_sdk_providers::streaming::{StreamAccumulator, StreamDelta, StreamErrorKind};
use agent_sdk_providers::{
    ChatOutcome, ChatRequest, ContentBlock, LlmProvider, Message, OpenAIApiSurface, OpenAIProvider,
    OpenAIReasoningConfig, OpenAIReasoningEffort, OpenAIReasoningSummary, OpenAIResponsesCompat,
    OpenAIResponsesProvider, ResponseFormat, SpeedTier, StopReason, Tool,
};
use anyhow::{Context, Result, bail, ensure};
use futures::StreamExt;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, matchers};

const MODEL: &str = "accounts/fireworks/models/kimi-k2p5";

type Bodies = Arc<Mutex<Vec<serde_json::Value>>>;

fn cassette(name: &str) -> Result<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(name);
    std::fs::read_to_string(&path).with_context(|| format!("read cassette {}", path.display()))
}

fn json_reply(name: &str) -> Result<ResponseTemplate> {
    let body: serde_json::Value = serde_json::from_str(&cassette(name)?)?;
    Ok(ResponseTemplate::new(200).set_body_json(body))
}

fn sse_reply(name: &str) -> Result<ResponseTemplate> {
    Ok(ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(cassette(name)?))
}

/// Serves the replies in order and keeps every request body.
struct Turns {
    replies: Mutex<VecDeque<ResponseTemplate>>,
    bodies: Bodies,
}

impl Respond for Turns {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        if let (Ok(mut bodies), Ok(body)) = (
            self.bodies.lock(),
            serde_json::from_slice::<serde_json::Value>(&request.body),
        ) {
            bodies.push(body);
        }
        self.replies
            .lock()
            .ok()
            .and_then(|mut replies| replies.pop_front())
            .unwrap_or_else(|| ResponseTemplate::new(500).set_body_string("no reply left"))
    }
}

async fn fireworks(replies: Vec<ResponseTemplate>) -> (MockServer, String, Bodies) {
    let server = MockServer::start().await;
    let bodies = Arc::new(Mutex::new(Vec::new()));
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/inference/v1/responses"))
        .respond_with(Turns {
            replies: Mutex::new(replies.into()),
            bodies: Arc::clone(&bodies),
        })
        .mount(&server)
        .await;
    let base_url = format!("{}/inference/v1", server.uri());
    (server, base_url, bodies)
}

fn provider(base_url: String) -> OpenAIResponsesProvider {
    OpenAIResponsesProvider::with_base_url("fw-test-key".to_owned(), MODEL.to_owned(), base_url)
        .with_compat(OpenAIResponsesCompat::FIREWORKS)
        .with_zero_retention(true)
}

fn sent(bodies: &Bodies) -> Result<Vec<serde_json::Value>> {
    Ok(bodies.lock().ok().context("bodies lock poisoned")?.clone())
}

fn balance_tool() -> Tool {
    Tool {
        name: "get_balance".to_owned(),
        description: "Read an account balance".to_owned(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {"account": {"type": "string"}},
            "required": ["account"],
            "additionalProperties": false
        }),
        display_name: "Balance".to_owned(),
        tier: agent_sdk_foundation::ToolTier::Observe,
        concurrency: agent_sdk_foundation::ToolConcurrency::ReadOnly,
        replay: agent_sdk_foundation::ToolReplay::Safe,
    }
}

fn first_turn() -> ChatRequest {
    ChatRequest::new(
        "You are a bookkeeping assistant.",
        vec![Message::user("What are my checking and savings balances?")],
    )
    .with_tools(vec![balance_tool()])
    .with_session_id("thread-fw-1")
}

fn second_turn(assistant: Vec<ContentBlock>) -> ChatRequest {
    let mut request = first_turn();
    request.messages.extend([
        Message::assistant_with_content(assistant),
        Message::tool_result("call_fw_1", "1200.00", false),
        Message::tool_result("call_fw_2", "300.00", false),
    ]);
    request
}

fn plain_reasoning(text: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}]
    })
}

fn tool_uses(blocks: &[ContentBlock]) -> Vec<(String, serde_json::Value)> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, input, .. } => Some((id.clone(), input.clone())),
            _ => None,
        })
        .collect()
}

fn thinking(blocks: &[ContentBlock]) -> Vec<&str> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        })
        .collect()
}

fn assert_stateless(body: &serde_json::Value) {
    assert_eq!(body["store"], false, "{body}");
    for field in [
        "previous_response_id",
        "include",
        "safety_identifier",
        "service_tier",
        "prompt_cache_options",
        "background",
    ] {
        assert!(body.get(field).is_none(), "{field} was sent: {body}");
    }
    assert!(
        !body.to_string().contains("\"phase\""),
        "message phase was sent: {body}"
    );
}

/// The second turn's input after system and user: the carried reasoning, the
/// two calls, then their outputs.
fn assert_carried_history(body: &serde_json::Value, reasoning: &str) -> Result<()> {
    let input = body["input"].as_array().context("input is not an array")?;
    ensure!(input.len() == 7, "unexpected input: {body}");
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[1]["role"], "user");
    assert_eq!(input[2], plain_reasoning(reasoning));
    assert_eq!(input[3]["type"], "function_call");
    assert_eq!(input[3]["call_id"], "call_fw_1");
    assert_eq!(input[3]["arguments"], r#"{"account":"checking"}"#);
    assert_eq!(input[4]["type"], "function_call");
    assert_eq!(input[4]["call_id"], "call_fw_2");
    assert_eq!(
        input[5],
        serde_json::json!({"type": "function_call_output", "call_id": "call_fw_1", "output": "1200.00"})
    );
    assert_eq!(
        input[6],
        serde_json::json!({"type": "function_call_output", "call_id": "call_fw_2", "output": "300.00"})
    );
    Ok(())
}

async fn drain(provider: &dyn LlmProvider, request: ChatRequest) -> Result<StreamAccumulator> {
    let mut stream = std::pin::pin!(provider.chat_stream(request));
    let mut accumulator = StreamAccumulator::new();
    while let Some(delta) = stream.next().await {
        let delta = delta?;
        if let StreamDelta::Error { message, kind } = &delta {
            bail!("stream error {kind:?}: {message}");
        }
        accumulator.apply(&delta);
    }
    Ok(accumulator)
}

#[tokio::test]
async fn tool_loop_carries_plain_reasoning_and_tool_outputs_statelessly() -> Result<()> {
    let (_server, base_url, bodies) = fireworks(vec![
        json_reply("fireworks_responses_tool_call.json")?,
        json_reply("fireworks_responses_final.json")?,
    ])
    .await;
    let provider = provider(base_url)
        .with_reasoning(
            OpenAIReasoningConfig::new()
                .with_effort(OpenAIReasoningEffort::High)
                .with_summary(OpenAIReasoningSummary::Auto)
                .with_safety_identifier("user-hash"),
        )
        .with_speed(SpeedTier::Fast);

    let ChatOutcome::Success(first) = provider.chat(first_turn()).await? else {
        bail!("the tool turn did not succeed");
    };
    assert_eq!(first.stop_reason, Some(StopReason::ToolUse));
    assert!(first.id.is_empty(), "a store=false response has no id");
    assert_eq!(
        tool_uses(&first.content),
        vec![
            (
                "call_fw_1".to_owned(),
                serde_json::json!({"account": "checking"})
            ),
            (
                "call_fw_2".to_owned(),
                serde_json::json!({"account": "savings"})
            ),
        ]
    );
    assert_eq!(
        thinking(&first.content),
        vec!["The user wants two balances; look both up at once."]
    );

    let ChatOutcome::Success(last) = provider.chat(second_turn(first.content)).await? else {
        bail!("the answer turn did not succeed");
    };
    assert_eq!(last.stop_reason, Some(StopReason::EndTurn));
    assert!(last.content.iter().any(|block| matches!(
        block,
        ContentBlock::Text { text } if text == "Checking holds 1200.00 and savings holds 300.00."
    )));

    let bodies = sent(&bodies)?;
    ensure!(
        bodies.len() == 2,
        "expected two requests, got {}",
        bodies.len()
    );
    for body in &bodies {
        assert_stateless(body);
        assert_eq!(body["model"], MODEL);
        assert_eq!(body["reasoning"], serde_json::json!({"effort": "high"}));
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["tools"][0]["name"], "get_balance");
        assert_eq!(body["prompt_cache_key"], "thread-fw-1");
    }
    assert_carried_history(
        &bodies[1],
        "The user wants two balances; look both up at once.",
    )
}

#[tokio::test]
async fn streamed_tool_loop_keeps_reasoning_for_the_next_turn() -> Result<()> {
    for cassette in [
        "fireworks_responses_stream_tool_call.sse",
        "fireworks_responses_stream_tool_call_without_item_ids.sse",
    ] {
        let (_server, base_url, bodies) = fireworks(vec![
            sse_reply(cassette)?,
            json_reply("fireworks_responses_final.json")?,
        ])
        .await;
        let provider = provider(base_url);

        let accumulator = drain(&provider, first_turn()).await?;
        assert_eq!(accumulator.stop_reason(), Some(&StopReason::ToolUse));
        let blocks = accumulator.into_content_blocks();
        assert_eq!(
            thinking(&blocks),
            vec!["Two lookups, in parallel."],
            "{cassette}"
        );
        assert_eq!(
            tool_uses(&blocks),
            vec![
                (
                    "call_fw_1".to_owned(),
                    serde_json::json!({"account": "checking"})
                ),
                (
                    "call_fw_2".to_owned(),
                    serde_json::json!({"account": "savings"})
                ),
            ],
            "{cassette}"
        );

        let outcome = provider.chat(second_turn(blocks)).await?;
        ensure!(
            matches!(outcome, ChatOutcome::Success(_)),
            "{cassette}: {outcome:?}"
        );
        let bodies = sent(&bodies)?;
        ensure!(bodies.len() == 2, "{cassette}: expected two requests");
        assert_eq!(bodies[0]["stream"], true);
        bodies.iter().for_each(assert_stateless);
        assert_carried_history(&bodies[1], "Two lookups, in parallel.")?;
    }
    Ok(())
}

#[tokio::test]
async fn structured_output_goes_as_a_strict_json_schema_text_format() -> Result<()> {
    let (_server, base_url, bodies) =
        fireworks(vec![json_reply("fireworks_responses_structured.json")?]).await;
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"payee": {"type": "string"}, "amount": {"type": "string"}},
        "required": ["payee", "amount"],
        "additionalProperties": false
    });
    let request = ChatRequest::new(
        String::new(),
        vec![Message::user(
            "Extract the payee and amount: Acme Supplies, 450.00",
        )],
    )
    .with_response_format(ResponseFormat::new("payment", schema.clone()));

    let ChatOutcome::Success(response) = provider(base_url).chat(request).await? else {
        bail!("the structured turn did not succeed");
    };
    assert!(response.content.iter().any(|block| matches!(
        block,
        ContentBlock::Text { text } if text == r#"{"payee":"Acme Supplies","amount":"450.00"}"#
    )));

    let bodies = sent(&bodies)?;
    let body = bodies.first().context("no request recorded")?;
    assert_stateless(body);
    assert_eq!(
        body["text"]["format"],
        serde_json::json!({"type": "json_schema", "name": "payment", "schema": schema, "strict": true})
    );
    Ok(())
}

#[tokio::test]
async fn zero_retention_refuses_store_true_before_any_request() -> Result<()> {
    let (server, base_url, _bodies) = fireworks(Vec::new()).await;
    let storing = OpenAIReasoningConfig::new().with_store(true);
    let responses = provider(base_url.clone()).with_reasoning(storing.clone());
    let rerouted = OpenAIProvider::with_base_url("fw-test-key", MODEL, base_url)
        .with_responses_compat(OpenAIResponsesCompat::FIREWORKS)
        .with_reasoning(storing.with_api_surface(OpenAIApiSurface::Responses))
        .with_zero_retention(true);

    for provider in [&responses as &dyn LlmProvider, &rerouted] {
        let outcome = provider.chat(first_turn()).await?;
        ensure!(
            matches!(&outcome, ChatOutcome::InvalidRequest(message) if message.contains("store=true")),
            "chat was not refused: {outcome:?}"
        );
        let mut stream = std::pin::pin!(provider.chat_stream(first_turn()));
        let first = stream.next().await.context("the stream ended silently")??;
        ensure!(
            matches!(
                &first,
                StreamDelta::Error { kind: StreamErrorKind::InvalidRequest, message }
                    if message.contains("store=true")
            ),
            "chat_stream was not refused: {first:?}"
        );
    }

    let received = server
        .received_requests()
        .await
        .context("mock server did not record requests")?;
    assert!(
        received.is_empty(),
        "a refused request reached the endpoint"
    );
    Ok(())
}

#[tokio::test]
async fn an_openai_compatible_provider_reroutes_to_fireworks_responses() -> Result<()> {
    let (_server, base_url, bodies) = fireworks(vec![
        json_reply("fireworks_responses_tool_call.json")?,
        sse_reply("fireworks_responses_stream_tool_call.sse")?,
    ])
    .await;
    let provider = OpenAIProvider::with_base_url("fw-test-key", MODEL, base_url)
        .with_responses_compat(OpenAIResponsesCompat::FIREWORKS)
        .with_reasoning(
            OpenAIReasoningConfig::new()
                .with_api_surface(OpenAIApiSurface::Responses)
                .with_store(false)
                .with_effort(OpenAIReasoningEffort::Medium),
        )
        .with_zero_retention(true);

    let ChatOutcome::Success(response) = provider.chat(first_turn()).await? else {
        bail!("the rerouted turn did not succeed");
    };
    assert_eq!(tool_uses(&response.content).len(), 2);
    let accumulator = drain(&provider, first_turn()).await?;
    assert_eq!(tool_uses(&accumulator.into_content_blocks()).len(), 2);

    let bodies = sent(&bodies)?;
    ensure!(bodies.len() == 2, "expected two requests");
    for body in &bodies {
        assert_stateless(body);
        assert_eq!(body["reasoning"], serde_json::json!({"effort": "medium"}));
    }
    Ok(())
}

fn live_fireworks_key() -> Option<String> {
    if std::env::var("AGENT_SDK_LIVE_SMOKE").ok().as_deref() != Some("1") {
        return None;
    }
    std::env::var("FIREWORKS_API_KEY")
        .ok()
        .filter(|key| !key.is_empty())
}

#[tokio::test]
#[ignore = "Tier C live smoke: set AGENT_SDK_LIVE_SMOKE=1 + FIREWORKS_API_KEY (synthetic prompts only)"]
async fn tier_c_live_fireworks_stateless_tool_loop() -> Result<()> {
    let Some(key) = live_fireworks_key() else {
        eprintln!("Tier C skipped: AGENT_SDK_LIVE_SMOKE/FIREWORKS_API_KEY not set");
        return Ok(());
    };
    let model = std::env::var("AGENT_SDK_FIREWORKS_MODEL").unwrap_or_else(|_| MODEL.to_owned());
    let provider = OpenAIResponsesProvider::fireworks(key, model)
        .with_reasoning(OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::Low));

    let accumulator = drain(&provider, first_turn()).await?;
    let blocks = accumulator.into_content_blocks();
    let calls = tool_uses(&blocks);
    ensure!(!calls.is_empty(), "expected a tool call, got {blocks:?}");

    let mut request = first_turn();
    request
        .messages
        .push(Message::assistant_with_content(blocks));
    request.messages.extend(
        calls
            .iter()
            .map(|(id, _)| Message::tool_result(id.clone(), "100.00", false)),
    );
    let outcome = provider.chat(request).await?;
    ensure!(
        matches!(outcome, ChatOutcome::Success(_)),
        "the stateless follow-up turn failed: {outcome:?}"
    );
    Ok(())
}
