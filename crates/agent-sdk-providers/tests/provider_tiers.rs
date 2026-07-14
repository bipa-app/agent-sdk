//! Tiered provider tests (plan §A2.4 / §6.2).
//!
//! Three tiers, run by different CI jobs:
//!
//! * **Tier A** — unit-level decode/round-trip checks that need no
//!   network at all: drive the trait's [`StreamAccumulator`] over
//!   hand-built [`StreamDelta`] sequences and assert the synthesized
//!   content (SSE-decode shape + `tool_use` JSON round-trip).
//! * **Tier B** — deterministic *integration* against a local
//!   [`wiremock`] server that replays scripted / recorded-and-redacted
//!   SSE cassettes.  Exercises the real `AnthropicProvider::chat_stream`
//!   HTTP + SSE-decode path end-to-end with **zero live network** and
//!   **zero API keys**.  Runs on every PR.
//! * **Tier C** — env-gated live smoke (one completion + one streamed
//!   turn + one tool call) against the real provider, only when
//!   `AGENT_SDK_LIVE_SMOKE=1` and a key are present.  Nightly/secrets
//!   only; `#[ignore]`d so a normal `nextest run` never touches it.
//!
//! `rvcr` (the cassette library named in the plan) targets `reqwest
//! 0.11` + `reqwest-middleware`; the SDK's providers are on `reqwest
//! 0.13` with a plain `reqwest::Client` that takes no middleware, so
//! rvcr cannot wrap them without pulling a second, unusable reqwest
//! into the tree.  The recorded/redacted/replayed *cassette* semantics
//! it would provide are implemented here on top of `wiremock` instead:
//! the `cassettes/` SSE fixtures are recorded-shape, secret-free, and
//! replayed deterministically.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use agent_sdk_providers::streaming::{StreamAccumulator, StreamDelta, StreamErrorKind};
use agent_sdk_providers::{AnthropicProvider, LlmProvider};
use agent_sdk_providers::{ChatRequest, ContentBlock, Message, StopReason};
use futures::StreamExt;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, matchers};

// ─────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────

/// Build a `ChatRequest` from a user prompt, optional tools, and a token
/// budget.  `ChatRequest` has no `Default`, so this keeps the call sites
/// terse without spelling out every field each time.
fn request(
    prompt: &str,
    tools: Option<Vec<agent_sdk_providers::Tool>>,
    max_tokens: u32,
) -> ChatRequest {
    ChatRequest {
        system: String::new(),
        messages: vec![Message::user(prompt)],
        tools,
        max_tokens,
        max_tokens_explicit: max_tokens != 0,
        session_id: None,
        cached_content: None,
        thinking: None,
        tool_choice: None,
        response_format: None,
        cache: None,
    }
}

/// A minimal one-message request; the body is irrelevant to the mock,
/// which replays a fixed cassette regardless of input.
fn sample_request() -> ChatRequest {
    request("hello", None, 1024)
}

/// Drain a provider stream into the accumulated content blocks plus the
/// ordered list of raw deltas, surfacing the first [`StreamDelta::Error`]
/// as an `Err` so error-injection cassettes are easy to assert on.
async fn drain(
    provider: &dyn LlmProvider,
    request: ChatRequest,
) -> Result<(Vec<ContentBlock>, Vec<StreamDelta>, Option<StopReason>), (StreamErrorKind, String)> {
    let mut stream = std::pin::pin!(provider.chat_stream(request));
    let mut acc = StreamAccumulator::new();
    let mut raw = Vec::new();
    while let Some(item) = stream.next().await {
        let delta = item.expect("transport-level stream error in deterministic test");
        if let StreamDelta::Error { kind, message } = &delta {
            return Err((*kind, message.clone()));
        }
        acc.apply(&delta);
        raw.push(delta);
    }
    let stop = acc.stop_reason().copied();
    Ok((acc.into_content_blocks(), raw, stop))
}

// ─────────────────────────────────────────────────────────────────────
// Tier A — unit: SSE-decode shape + tool_use JSON round-trip
// ─────────────────────────────────────────────────────────────────────

#[test]
fn tier_a_text_deltas_accumulate_in_block_order() {
    let mut acc = StreamAccumulator::new();
    for delta in [
        StreamDelta::TextDelta {
            delta: "Hel".into(),
            block_index: 0,
        },
        StreamDelta::TextDelta {
            delta: "lo".into(),
            block_index: 0,
        },
        StreamDelta::Done {
            stop_reason: Some(StopReason::EndTurn),
        },
    ] {
        acc.apply(&delta);
    }
    assert_eq!(acc.stop_reason(), Some(&StopReason::EndTurn));
    let blocks = acc.into_content_blocks();
    assert_eq!(blocks.len(), 1);
    assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello"));
}

#[test]
fn tier_a_tool_use_json_round_trips_through_streamed_fragments() {
    // The provider streams partial JSON for tool inputs; the
    // accumulator must concatenate the fragments and parse the final
    // JSON back into the original object.
    let mut acc = StreamAccumulator::new();
    let fragments = [r#"{"comm"#, r#"and":"#, r#""ls -la""#, "}"];
    acc.apply(&StreamDelta::ToolUseStart {
        id: "toolu_1".into(),
        name: "bash".into(),
        block_index: 0,
        thought_signature: None,
    });
    for frag in fragments {
        acc.apply(&StreamDelta::ToolInputDelta {
            id: "toolu_1".into(),
            delta: frag.into(),
            block_index: 0,
        });
    }
    acc.apply(&StreamDelta::Done {
        stop_reason: Some(StopReason::ToolUse),
    });

    let blocks = acc.into_content_blocks();
    assert_eq!(blocks.len(), 1);
    let ContentBlock::ToolUse {
        id, name, input, ..
    } = &blocks[0]
    else {
        panic!("expected ToolUse, got {:?}", blocks[0]);
    };
    assert_eq!(id, "toolu_1");
    assert_eq!(name, "bash");
    assert_eq!(input, &serde_json::json!({"command": "ls -la"}));
}

#[test]
fn tier_a_interleaved_text_and_tool_use_sort_by_block_index() {
    // Out-of-order arrival (tool block before its text neighbour) must
    // still sort into block order in the synthesized content.
    let mut acc = StreamAccumulator::new();
    acc.apply(&StreamDelta::ToolUseStart {
        id: "toolu_x".into(),
        name: "search".into(),
        block_index: 1,
        thought_signature: None,
    });
    acc.apply(&StreamDelta::TextDelta {
        delta: "Let me check.".into(),
        block_index: 0,
    });
    acc.apply(&StreamDelta::ToolInputDelta {
        id: "toolu_x".into(),
        delta: r#"{"q":"rust"}"#.into(),
        block_index: 1,
    });
    let blocks = acc.into_content_blocks();
    assert_eq!(blocks.len(), 2);
    assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Let me check."));
    assert!(matches!(&blocks[1], ContentBlock::ToolUse { name, .. } if name == "search"));
}

// ─────────────────────────────────────────────────────────────────────
// Tier B — deterministic integration: scripted/cassette SSE via wiremock
// ─────────────────────────────────────────────────────────────────────

/// A `wiremock` responder that streams a fixed SSE body.  `wiremock`
/// returns the whole body at once, which the provider's chunked SSE
/// decoder handles identically to a multi-chunk stream (it buffers and
/// splits on blank lines).  Recording the *request* it saw lets tests
/// assert the provider posted to `/v1/messages`.
struct SseCassette {
    body: String,
    seen: Arc<Mutex<Vec<String>>>,
}

impl Respond for SseCassette {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        if let Ok(mut seen) = self.seen.lock() {
            seen.push(String::from_utf8_lossy(&request.body).into_owned());
        }
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(self.body.clone())
    }
}

/// Load a recorded/redacted SSE cassette from `tests/cassettes/`.
fn cassette(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cassettes")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read cassette {}: {e}", path.display()))
}

/// Stand up a wiremock server that replays `body` for `POST
/// /v1/messages`, and an `AnthropicProvider` pointed at it.
async fn provider_for(body: String) -> (MockServer, AnthropicProvider, Arc<Mutex<Vec<String>>>) {
    let server = MockServer::start().await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/messages"))
        .respond_with(SseCassette {
            body,
            seen: Arc::clone(&seen),
        })
        .mount(&server)
        .await;
    let provider =
        AnthropicProvider::new("test-key-not-a-secret", "claude-test").with_base_url(server.uri());
    (server, provider, seen)
}

#[tokio::test]
async fn tier_b_text_cassette_decodes_to_text_block() {
    let (_server, provider, seen) = provider_for(cassette("anthropic_text.sse")).await;
    let (blocks, _raw, stop) = drain(&provider, sample_request())
        .await
        .expect("text cassette must not error");

    assert_eq!(stop, Some(StopReason::EndTurn));
    assert_eq!(blocks.len(), 1);
    assert!(
        matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello, world!"),
        "decoded {blocks:?}",
    );
    // The provider actually issued the POST to the mocked endpoint.
    assert_eq!(seen.lock().expect("seen lock").len(), 1);
}

#[tokio::test]
async fn tier_b_tool_use_cassette_round_trips_json() {
    let (_server, provider, _seen) = provider_for(cassette("anthropic_tool_use.sse")).await;
    let (blocks, _raw, stop) = drain(&provider, sample_request())
        .await
        .expect("tool cassette must not error");

    assert_eq!(stop, Some(StopReason::ToolUse));
    let tool = blocks
        .iter()
        .find_map(|b| match b {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Some((id, name, input)),
            _ => None,
        })
        .expect("a ToolUse block");
    assert_eq!(tool.0, "toolu_01ABC");
    assert_eq!(tool.1, "get_weather");
    assert_eq!(tool.2, &serde_json::json!({"location": "Lisbon"}));
}

#[tokio::test]
async fn tier_b_interleaved_text_and_tool_use_cassette_orders_blocks() {
    let (_server, provider, _seen) = provider_for(cassette("anthropic_text_and_tool.sse")).await;
    let (blocks, _raw, stop) = drain(&provider, sample_request())
        .await
        .expect("interleaved cassette must not error");

    assert_eq!(stop, Some(StopReason::ToolUse));
    assert_eq!(blocks.len(), 2, "one text + one tool block: {blocks:?}");
    assert!(matches!(&blocks[0], ContentBlock::Text { text } if text.contains("weather")));
    assert!(matches!(&blocks[1], ContentBlock::ToolUse { name, .. } if name == "get_weather"));
}

#[tokio::test]
async fn tier_b_server_error_surfaces_recoverable_stream_error() {
    // An HTTP 503 from the endpoint must surface as a recoverable
    // `ServerError` stream delta (the worker's retry wrapper keys off
    // this), not a transport panic.
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/messages"))
        .respond_with(ResponseTemplate::new(503).set_body_string("upstream unavailable"))
        .mount(&server)
        .await;
    let provider =
        AnthropicProvider::new("test-key-not-a-secret", "claude-test").with_base_url(server.uri());

    let err = drain(&provider, sample_request())
        .await
        .expect_err("503 must surface as a stream error");
    assert_eq!(err.0, StreamErrorKind::ServerError);
}

#[tokio::test]
async fn tier_b_rate_limit_surfaces_recoverable_stream_error() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .mount(&server)
        .await;
    let provider =
        AnthropicProvider::new("test-key-not-a-secret", "claude-test").with_base_url(server.uri());

    let err = drain(&provider, sample_request())
        .await
        .expect_err("429 must surface as a stream error");
    // No `Retry-After` on the response, so the stream error carries no hint
    // and the caller falls back to its own backoff.
    assert_eq!(err.0, StreamErrorKind::RateLimited(None));
    assert!(err.0.is_recoverable());
}

#[tokio::test]
async fn tier_b_rate_limit_stream_error_carries_retry_after_header() {
    // The `Retry-After` on a streaming 429 must survive the stream error
    // channel, so the streaming retry path honours the server's delay
    // exactly like the non-streaming `chat()` path does.
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "30")
                .set_body_string("slow down"),
        )
        .mount(&server)
        .await;
    let provider =
        AnthropicProvider::new("test-key-not-a-secret", "claude-test").with_base_url(server.uri());

    let err = drain(&provider, sample_request())
        .await
        .expect_err("429 must surface as a stream error");
    assert_eq!(
        err.0,
        StreamErrorKind::RateLimited(Some(Duration::from_secs(30)))
    );
    assert_eq!(err.0.retry_after(), Some(Duration::from_secs(30)));
}

#[tokio::test]
async fn tier_b_invalid_request_surfaces_fatal_stream_error() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad schema"))
        .mount(&server)
        .await;
    let provider =
        AnthropicProvider::new("test-key-not-a-secret", "claude-test").with_base_url(server.uri());

    let err = drain(&provider, sample_request())
        .await
        .expect_err("400 must surface as a stream error");
    assert_eq!(err.0, StreamErrorKind::InvalidRequest);
    assert!(!err.0.is_recoverable());
}

#[tokio::test]
async fn tier_b_stream_truncated_after_tool_start_surfaces_recoverable_error() {
    // A cassette that ends after `content_block_start` for a tool use
    // but *before* `message_stop` — the real Anthropic decoder treats a
    // missing terminal event as a recoverable `ServerError` so the
    // worker's retry wrapper re-attempts (rather than committing a
    // half-finished turn).  The deltas that *did* arrive before the cut
    // must still be well-formed: a `ToolUseStart` with a real id/name,
    // never a half-parsed fragment.
    let (_server, provider, _seen) = provider_for(cassette("anthropic_tool_truncated.sse")).await;

    let mut stream = std::pin::pin!(provider.chat_stream(sample_request()));
    let mut pre_error = Vec::new();
    let mut error_kind = None;
    while let Some(item) = stream.next().await {
        match item.expect("no transport error in deterministic test") {
            StreamDelta::Error { kind, .. } => {
                error_kind = Some(kind);
                break;
            }
            other => pre_error.push(other),
        }
    }

    assert_eq!(
        error_kind,
        Some(StreamErrorKind::ServerError),
        "a stream cut before message_stop is a recoverable ServerError",
    );
    assert!(error_kind.is_some_and(StreamErrorKind::is_recoverable));

    // The one delta that arrived is a well-formed ToolUseStart.
    let started = pre_error
        .iter()
        .find_map(|d| match d {
            StreamDelta::ToolUseStart { id, name, .. } => Some((id, name)),
            _ => None,
        })
        .expect("the ToolUseStart that arrived before the cut");
    assert_eq!(started.0, "toolu_03GHI");
    assert_eq!(started.1, "get_weather");
}

// ─────────────────────────────────────────────────────────────────────
// Tier C — env-gated live smoke (nightly/secrets only)
// ─────────────────────────────────────────────────────────────────────
//
// These are `#[ignore]`d so a default `nextest run` skips them, AND
// additionally gated on `AGENT_SDK_LIVE_SMOKE=1` + a real key so even
// `--run-ignored all` is a no-op without explicit opt-in.  The nightly
// secrets CI job sets both.

/// `true` only when live smoke is explicitly enabled and a key exists.
fn live_smoke_enabled() -> Option<String> {
    if std::env::var("AGENT_SDK_LIVE_SMOKE").ok().as_deref() != Some("1") {
        return None;
    }
    std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
}

#[tokio::test]
#[ignore = "Tier C live smoke: set AGENT_SDK_LIVE_SMOKE=1 + ANTHROPIC_API_KEY (nightly/secrets only)"]
async fn tier_c_live_completion_and_stream_and_tool() -> anyhow::Result<()> {
    let Some(key) = live_smoke_enabled() else {
        eprintln!("Tier C skipped: AGENT_SDK_LIVE_SMOKE/ANTHROPIC_API_KEY not set");
        return Ok(());
    };
    let model =
        std::env::var("AGENT_SDK_LIVE_MODEL").unwrap_or_else(|_| "claude-haiku-4-5".to_owned());
    let provider = AnthropicProvider::new(key, model);

    // 1) One streamed text turn.
    let (blocks, _raw, stop) = drain(&provider, request("Reply with exactly: pong", None, 64))
        .await
        .map_err(|(kind, msg)| anyhow::anyhow!("live stream error ({kind:?}): {msg}"))?;
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::Text { .. })),
        "expected at least one text block from a live turn",
    );
    assert!(stop.is_some(), "live turn must report a stop reason");

    // 2) One non-streaming completion.
    let outcome = provider
        .chat(request("Say hi in one word.", None, 32))
        .await?;
    assert!(
        matches!(outcome, agent_sdk_providers::ChatOutcome::Success(_)),
        "live completion should succeed",
    );

    // 3) One tool call.
    let weather_tool = agent_sdk_providers::Tool {
        name: "get_weather".into(),
        description: "Get the weather for a city".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
        display_name: "Weather".into(),
        tier: agent_sdk_foundation::ToolTier::Observe,
    };
    let (tool_blocks, _raw, _stop) = drain(
        &provider,
        request(
            "What's the weather in Lisbon? Use the tool.",
            Some(vec![weather_tool]),
            256,
        ),
    )
    .await
    .map_err(|(kind, msg)| anyhow::anyhow!("live tool stream error ({kind:?}): {msg}"))?;
    assert!(
        tool_blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. })),
        "expected a tool_use block when a tool is offered",
    );

    Ok(())
}
