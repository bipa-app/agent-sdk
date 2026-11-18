//! Tier C live smoke for the Bedrock provider.
//!
//! `#[ignore]`d, and a no-op unless `AGENT_SDK_LIVE_SMOKE=1` and the Bedrock
//! environment below are set, so `--run-ignored all` without credentials
//! does nothing.
//!
//! * `AGENT_SDK_BEDROCK_REGION`: the runtime region.
//! * `AGENT_SDK_BEDROCK_PROFILE` (`us`, `global`, ...) or
//!   `AGENT_SDK_BEDROCK_MODEL_ID`: how the model is addressed.
//! * `AGENT_SDK_BEDROCK_MODEL`: the Anthropic model id (default
//!   `claude-sonnet-5-5`).
//! * Credentials: what [`BedrockAuth::from_env`] reads, or
//!   `AGENT_SDK_BEDROCK_ROLE_ARN` with `AGENT_SDK_BEDROCK_WEB_IDENTITY_AUDIENCE`
//!   to assume a role with a Google ID token from the GCE metadata server.
//! * `ANTHROPIC_API_KEY` (optional): enables the thinking-replay check, which
//!   sends a thinking block made on the Anthropic API to Bedrock.

#![cfg(feature = "bedrock")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_sdk_foundation::llm::{
    CacheConfig, CacheTtl, ChatOutcome, ChatRequest, Content, ContentBlock, ContentSource, Effort,
    Message, ResponseFormat, Role, ThinkingConfig, ThinkingDisplay, ToolChoice,
};
use agent_sdk_providers::LlmProvider;
use agent_sdk_providers::impls::AnthropicProvider;
use agent_sdk_providers::impls::bedrock::{
    BedrockAuth, BedrockInferenceProfile, BedrockProvider, PrefixMismatchBehavior,
    WebIdentityCredentials, WebIdentityToken, bedrock_inference_profile_id,
};
use agent_sdk_providers::streaming::{StreamAccumulator, StreamDelta, StreamErrorKind};
use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use futures::StreamExt;

struct Live {
    auth: BedrockAuth,
    region: String,
    model: String,
    model_id: String,
}

impl Live {
    fn provider(&self) -> BedrockProvider {
        BedrockProvider::new(
            self.auth.clone(),
            self.region.clone(),
            self.model.clone(),
            self.model_id.clone(),
        )
    }

    fn provider_for(&self, model: &str, model_id: &str) -> BedrockProvider {
        BedrockProvider::new(self.auth.clone(), self.region.clone(), model, model_id)
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn profile(prefix: &str) -> Result<BedrockInferenceProfile> {
    use BedrockInferenceProfile::{Apac, Au, Ca, Eu, Global, In, Jp, Us};
    Ok(match prefix {
        "global" => Global,
        "us" => Us,
        "eu" => Eu,
        "apac" => Apac,
        "jp" => Jp,
        "au" => Au,
        "ca" => Ca,
        "in" => In,
        other => bail!("unknown inference profile prefix {other}"),
    })
}

fn live() -> Result<Option<Live>> {
    if env("AGENT_SDK_LIVE_SMOKE").as_deref() != Some("1") {
        return Ok(None);
    }
    let Some(region) = env("AGENT_SDK_BEDROCK_REGION") else {
        return Ok(None);
    };
    let model = env("AGENT_SDK_BEDROCK_MODEL").unwrap_or_else(|| "claude-sonnet-5-5".to_owned());
    let model_id = match (
        env("AGENT_SDK_BEDROCK_MODEL_ID"),
        env("AGENT_SDK_BEDROCK_PROFILE"),
    ) {
        (Some(model_id), _) => model_id,
        (None, Some(prefix)) => bedrock_inference_profile_id(&model, profile(&prefix)?)
            .with_context(|| format!("no documented {prefix} profile for {model}"))?,
        (None, None) => bail!("set AGENT_SDK_BEDROCK_PROFILE or AGENT_SDK_BEDROCK_MODEL_ID"),
    };
    let auth = match (
        env("AGENT_SDK_BEDROCK_ROLE_ARN"),
        env("AGENT_SDK_BEDROCK_WEB_IDENTITY_AUDIENCE"),
    ) {
        (Some(role_arn), Some(audience)) => {
            BedrockAuth::WebIdentity(Arc::new(WebIdentityCredentials::new(
                role_arn,
                "agent-sdk-live-smoke",
                WebIdentityToken::GcpMetadata { audience },
                &region,
            )))
        }
        _ => BedrockAuth::from_env(&region)?,
    };
    Ok(Some(Live {
        auth,
        region,
        model,
        model_id,
    }))
}

fn lookup_tool() -> agent_sdk_providers::Tool {
    agent_sdk_providers::Tool {
        name: "get_due_date".into(),
        description: "Look up the due date of an invoice by its number".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": { "invoice": { "type": "string" } },
            "required": ["invoice"],
        }),
        display_name: "Due date".into(),
        tier: agent_sdk_foundation::ToolTier::Observe,
        concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
        replay: agent_sdk_foundation::ToolReplay::Unsafe,
    }
}

/// Drain a stream, reporting the longest silence between deltas and how many
/// keep-alives arrived.
async fn drain(
    provider: &dyn LlmProvider,
    request: ChatRequest,
) -> Result<(StreamAccumulator, Duration, usize)> {
    let mut stream = provider.chat_stream(request);
    let mut accumulator = StreamAccumulator::new();
    let mut last = Instant::now();
    let mut longest_gap = Duration::ZERO;
    let mut keep_alives = 0;
    while let Some(delta) = stream.next().await {
        let delta = delta?;
        longest_gap = longest_gap.max(last.elapsed());
        last = Instant::now();
        match &delta {
            StreamDelta::Error { message, kind } => bail!("stream error ({kind:?}): {message}"),
            StreamDelta::KeepAlive => keep_alives += 1,
            _ => {}
        }
        accumulator.apply(&delta);
    }
    Ok((accumulator, longest_gap, keep_alives))
}

fn text_of(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// A one-page PDF whose page reads `text`, with a correct cross-reference
/// table.
fn pdf_with_text(text: &str) -> Vec<u8> {
    let stream = format!("BT /F1 24 Tf 72 720 Td ({text}) Tj ET");
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_owned(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
         /Resources << /Font << /F1 5 0 R >> >> >>"
            .to_owned(),
        format!(
            "<< /Length {} >>\nstream\n{stream}\nendstream",
            stream.len()
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
    ];
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (index, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", index + 1).as_bytes());
    }
    let xref = pdf.len();
    pdf.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
    );
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .as_bytes(),
    );
    pdf
}

#[tokio::test]
#[ignore = "Tier C live smoke: AGENT_SDK_LIVE_SMOKE=1 + Bedrock env (see module docs)"]
async fn tier_c_bedrock_completion_stream_and_tools() -> Result<()> {
    let Some(live) = live()? else {
        eprintln!(
            "Tier C Bedrock skipped: AGENT_SDK_LIVE_SMOKE / AGENT_SDK_BEDROCK_REGION not set"
        );
        return Ok(());
    };
    let provider = live.provider();

    let outcome = provider
        .chat(
            ChatRequest::new("", vec![Message::user("Reply with exactly: pong")])
                .with_max_tokens(256),
        )
        .await?;
    let ChatOutcome::Success(response) = outcome else {
        bail!("completion: {outcome:?}");
    };
    eprintln!(
        "completion: model={} usage={:?}",
        response.model, response.usage
    );
    assert!(text_of(&response.content).to_lowercase().contains("pong"));

    let (accumulator, gap, keep_alives) = drain(
        &provider,
        ChatRequest::new(
            "",
            vec![Message::user("Count from 1 to 20, one number per line.")],
        )
        .with_max_tokens(512),
    )
    .await?;
    eprintln!(
        "stream: route={:?} model={:?} usage={:?} longest_gap={gap:?} keep_alives={keep_alives}",
        accumulator.served_route(),
        accumulator.served_model(),
        accumulator.usage()
    );
    assert_eq!(accumulator.served_route(), Some("bedrock"));
    assert!(text_of(&accumulator.into_content_blocks()).contains("20"));

    let (accumulator, _, _) = drain(
        &provider,
        ChatRequest::new(
            "Use the tool to answer.",
            vec![Message::user("When is invoice 4471 due?")],
        )
        .with_tools(vec![lookup_tool()])
        .with_tool_choice(ToolChoice::Auto)
        .with_max_tokens(1024),
    )
    .await?;
    let blocks = accumulator.into_content_blocks();
    eprintln!("tool turn: {blocks:?}");
    assert!(blocks.iter().any(|block| matches!(
        block,
        ContentBlock::ToolUse { name, input, .. } if name == "get_due_date" && input["invoice"] == "4471"
    )));
    Ok(())
}

#[tokio::test]
#[ignore = "Tier C live smoke: AGENT_SDK_LIVE_SMOKE=1 + Bedrock env (see module docs)"]
async fn tier_c_bedrock_long_thinking_stream_keeps_its_connection() -> Result<()> {
    let Some(live) = live()? else {
        eprintln!("Tier C Bedrock skipped");
        return Ok(());
    };
    let provider = live
        .provider()
        .with_thinking(ThinkingConfig::adaptive_with_effort(Effort::High));
    let (accumulator, gap, keep_alives) = drain(
        &provider,
        ChatRequest::new(
            "",
            vec![Message::user(
                "Find every integer n between 1 and 400 for which n^2 + n + 41 is not prime. \
                 Check each candidate carefully before you answer, then list them.",
            )],
        )
        .with_max_tokens(32_000),
    )
    .await?;
    eprintln!(
        "long thinking: usage={:?} longest_gap={gap:?} keep_alives={keep_alives}",
        accumulator.usage()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "Tier C live smoke: AGENT_SDK_LIVE_SMOKE=1 + Bedrock env (see module docs)"]
async fn tier_c_bedrock_structured_output_cache_and_pdf() -> Result<()> {
    let Some(live) = live()? else {
        eprintln!("Tier C Bedrock skipped");
        return Ok(());
    };
    let provider = live
        .provider()
        .with_thinking(ThinkingConfig::adaptive_with_effort(Effort::Low));

    let format = ResponseFormat::new(
        "total",
        serde_json::json!({
            "type": "object",
            "properties": { "total": { "type": "integer" } },
            "required": ["total"],
            "additionalProperties": false
        }),
    );
    let outcome = provider
        .chat(
            ChatRequest::new("", vec![Message::user("Add 1234 and 4321.")])
                .with_response_format(format)
                .with_max_tokens(2048),
        )
        .await?;
    let ChatOutcome::Success(response) = outcome else {
        bail!("structured output: {outcome:?}");
    };
    let value: serde_json::Value = serde_json::from_str(&text_of(&response.content))?;
    assert_eq!(value["total"], 5555);

    // A system prompt past the 512-token cache minimum, cached for an hour.
    let system = "You are a careful reader of scanned receipts. ".repeat(200);
    let cached = || {
        ChatRequest::new(system.clone(), vec![Message::user("Say ok.")])
            .with_cache(CacheConfig::default().with_ttl(CacheTtl::OneHour))
            .with_max_tokens(256)
    };
    let first = provider.chat(cached()).await?;
    let second = provider.chat(cached()).await?;
    let (ChatOutcome::Success(first), ChatOutcome::Success(second)) = (first, second) else {
        bail!("cache calls failed");
    };
    eprintln!("cache: first={:?} second={:?}", first.usage, second.usage);
    assert!(
        first.usage.cache_creation_input_tokens + first.usage.cached_input_tokens > 0,
        "the first call neither wrote nor read the cache"
    );
    assert!(
        second.usage.cached_input_tokens > 0,
        "the second call read no cache"
    );

    let pdf = base64::engine::general_purpose::STANDARD.encode(pdf_with_text("Total due: 5555"));
    let outcome = provider
        .chat(
            ChatRequest::new(
                "",
                vec![Message {
                    role: Role::User,
                    content: Content::Blocks(vec![
                        ContentBlock::Document {
                            source: ContentSource::new("application/pdf", pdf),
                        },
                        ContentBlock::Text {
                            text:
                                "What total does this document state? Answer with the number only."
                                    .to_owned(),
                        },
                    ]),
                }],
            )
            .with_max_tokens(1024),
        )
        .await?;
    let ChatOutcome::Success(response) = outcome else {
        bail!("pdf: {outcome:?}");
    };
    assert!(text_of(&response.content).contains("5555"));
    Ok(())
}

#[tokio::test]
#[ignore = "Tier C live smoke: AGENT_SDK_LIVE_SMOKE=1 + Bedrock env (see module docs)"]
async fn tier_c_bedrock_errors_map_to_their_classes() -> Result<()> {
    let Some(live) = live()? else {
        eprintln!("Tier C Bedrock skipped");
        return Ok(());
    };
    let request = || ChatRequest::new("", vec![Message::user("hi")]).with_max_tokens(64);

    // A model id Bedrock does not know.
    let unknown = live.provider_for(&live.model, "us.anthropic.claude-not-a-model");
    let outcome = unknown.chat(request()).await?;
    eprintln!("unknown model id: {outcome:?}");
    assert!(matches!(outcome, ChatOutcome::InvalidRequest(_)));
    let mut stream = unknown.chat_stream(request());
    let delta = stream.next().await.context("no delta")??;
    eprintln!("unknown model id, streaming: {delta:?}");
    assert!(matches!(
        delta,
        StreamDelta::Error {
            kind: StreamErrorKind::InvalidRequest,
            ..
        }
    ));

    // The model card says Sonnet 5.5 takes tool_choice auto or none only.
    // The provider refuses a forced tool before the wire; this sends one past
    // that guard (an Anthropic id it has no table row for) to confirm Bedrock
    // refuses it too.
    if let Some(sonnet) =
        bedrock_inference_profile_id("claude-sonnet-5-5", BedrockInferenceProfile::Us)
    {
        let unguarded = live.provider_for("claude-sonnet-5-5-unlisted", &sonnet);
        let outcome = unguarded
            .chat(
                request()
                    .with_tools(vec![lookup_tool()])
                    .with_tool_choice(ToolChoice::Tool("get_due_date".to_owned())),
            )
            .await?;
        eprintln!("forced tool on Sonnet 5.5: {outcome:?}");
    }
    Ok(())
}

fn pay_tool() -> agent_sdk_providers::Tool {
    agent_sdk_providers::Tool {
        name: "pay_invoice".into(),
        description: "Pay an invoice by its number".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": { "invoice": { "type": "string" } },
            "required": ["invoice"],
        }),
        display_name: "Pay".into(),
        tier: agent_sdk_foundation::ToolTier::Confirm,
        concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
        replay: agent_sdk_foundation::ToolReplay::Unsafe,
    }
}

/// A harness that adds a tool after the model has thought (a skill loading
/// its tools) replays that thinking under a tool list it was not signed with.
/// Accounts Bedrock enforces preserved thinking on answer that with a 400.
#[tokio::test]
#[ignore = "Tier C live smoke: AGENT_SDK_LIVE_SMOKE=1 + Bedrock env (see module docs)"]
async fn tier_c_bedrock_thinking_survives_a_tool_list_change() -> Result<()> {
    let Some(live) = live()? else {
        eprintln!("Tier C Bedrock tool-list change skipped");
        return Ok(());
    };
    let thinking = ThinkingConfig::adaptive().with_effort(Effort::Max);
    let system = "Use the tools to answer.";
    let question = Message::user(
        "Invoice 4471 was issued on 2026-10-03 on net 45, invoice 4472 on 2026-09-30 on net 60, \
         and invoice 4473 on 2026-10-20 on net 30, counting calendar days. Before you call any \
         tool, work out by hand which one is due first. Then look that one up to confirm.",
    );
    let mut thought = None;
    for _ in 0..3 {
        let outcome = live
            .provider()
            .with_thinking(thinking.clone())
            .chat(
                ChatRequest::new(system, vec![question.clone()])
                    .with_tools(vec![lookup_tool()])
                    .with_max_tokens(16_384),
            )
            .await?;
        let ChatOutcome::Success(first) = outcome else {
            bail!("first turn: {outcome:?}");
        };
        if first
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Thinking { .. }))
        {
            thought = Some(first);
            break;
        }
    }
    let Some(first) = thought else {
        bail!("three first turns answered without thinking, so nothing was bound");
    };
    let thinking_blocks = first
        .content
        .iter()
        .filter(|block| matches!(block, ContentBlock::Thinking { .. }))
        .count();
    let tool_use_id = first
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .context("the first turn called no tool")?;
    let replay = ChatRequest::new(
        system,
        vec![
            question,
            Message {
                role: Role::Assistant,
                content: Content::Blocks(first.content.clone()),
            },
            Message::tool_result(tool_use_id, "2026-12-01", false),
        ],
    )
    .with_tools(vec![lookup_tool(), pay_tool()])
    .with_max_tokens(16_384);
    let refused = live
        .provider()
        .with_thinking(thinking.clone())
        .chat(replay.clone())
        .await?;
    eprintln!(
        "thinking blocks: {thinking_blocks}; replay with the default behavior: {}",
        outcome_summary(&refused)
    );
    let outcome = live
        .provider()
        .with_thinking(thinking)
        .with_prefix_mismatch(PrefixMismatchBehavior::DropBlock)
        .chat(replay)
        .await?;
    eprintln!("replay with drop_block: {}", outcome_summary(&outcome));
    let ChatOutcome::Success(_) = outcome else {
        bail!("the replay with drop_block failed: {outcome:?}");
    };
    Ok(())
}

/// One line per outcome: the error text, or the stop reason and the block kinds.
fn outcome_summary(outcome: &ChatOutcome) -> String {
    match outcome {
        ChatOutcome::Success(response) => format!(
            "Success stop={:?} blocks={:?}",
            response.stop_reason,
            response
                .content
                .iter()
                .map(|block| match block {
                    ContentBlock::Thinking { .. } => "thinking",
                    ContentBlock::RedactedThinking { .. } => "redacted_thinking",
                    ContentBlock::Text { .. } => "text",
                    ContentBlock::ToolUse { .. } => "tool_use",
                    _ => "other",
                })
                .collect::<Vec<_>>()
        ),
        other => format!("{other:?}"),
    }
}

/// Sends a thinking block made on the Anthropic API to Bedrock, the case of a
/// conversation that changes transport mid-thread. The Bedrock docs say a
/// block from another organization is dropped and the request succeeds; an
/// undecryptable signature is a 400.
#[tokio::test]
#[ignore = "Tier C live smoke: AGENT_SDK_LIVE_SMOKE=1 + Bedrock env + ANTHROPIC_API_KEY"]
async fn tier_c_bedrock_takes_a_thinking_block_from_the_anthropic_api() -> Result<()> {
    let (Some(live), Some(key)) = (live()?, env("ANTHROPIC_API_KEY")) else {
        eprintln!("Tier C Bedrock thinking replay skipped");
        return Ok(());
    };
    let thinking = ThinkingConfig::adaptive().with_display(ThinkingDisplay::Summarized);
    let system = "Use the tool to answer.";
    let question = Message::user("When is invoice 4471 due? Think it through first.");
    let direct = AnthropicProvider::new(key, live.model.clone()).with_thinking(thinking.clone());
    let outcome = direct
        .chat(
            ChatRequest::new(system, vec![question.clone()])
                .with_tools(vec![lookup_tool()])
                .with_max_tokens(4096),
        )
        .await?;
    let ChatOutcome::Success(first) = outcome else {
        bail!("direct API turn: {outcome:?}");
    };
    let tool_use_id = first
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .context("the direct API turn called no tool")?;
    let thinking_blocks = first
        .content
        .iter()
        .filter(|block| matches!(block, ContentBlock::Thinking { .. }))
        .count();
    let follow_up = ChatRequest::new(
        system,
        vec![
            question,
            Message {
                role: Role::Assistant,
                content: Content::Blocks(first.content.clone()),
            },
            Message::tool_result(tool_use_id, "2026-12-01", false),
        ],
    )
    .with_tools(vec![lookup_tool()])
    .with_max_tokens(4096);
    let outcome = live
        .provider()
        .with_thinking(thinking)
        .chat(follow_up)
        .await?;
    eprintln!("direct API thinking blocks: {thinking_blocks}; Bedrock replay: {outcome:?}");
    Ok(())
}

/// The way back: a thread that ran on Bedrock moves to the Anthropic API, as
/// a rollback of the transport does. The Anthropic API gets thinking blocks
/// that Bedrock's account signed, in two shapes: the last assistant turn of a
/// tool-use continuation (its `tool_use` still waits on the result), and a
/// finished exchange followed by a new question. The preserved-thinking docs
/// say a block from another account is dropped before the model sees it and
/// the request succeeds; both replays must answer.
///
/// The blocks go out with their recorded route cleared, as a block stored
/// before routes were recorded: with the route kept, the provider leaves the
/// blocks out or refuses the tool round itself, and the API would never see
/// them. This test measures the API. A failure in the tool round means the API
/// refuses another account's block in that position, which is the case the
/// provider's refusal of a routed block there exists for.
#[tokio::test]
#[ignore = "Tier C live smoke: AGENT_SDK_LIVE_SMOKE=1 + Bedrock env + ANTHROPIC_API_KEY"]
async fn tier_c_anthropic_api_takes_a_thinking_block_from_bedrock() -> Result<()> {
    let (Some(live), Some(key)) = (live()?, env("ANTHROPIC_API_KEY")) else {
        eprintln!("Tier C Anthropic API thinking replay skipped");
        return Ok(());
    };
    let thinking = ThinkingConfig::adaptive().with_display(ThinkingDisplay::Summarized);
    let system = "Use the tool to answer.";
    let question = Message::user("When is invoice 4471 due? Think it through first.");
    let bedrock = live.provider().with_thinking(thinking.clone());
    let outcome = bedrock
        .chat(
            ChatRequest::new(system, vec![question.clone()])
                .with_tools(vec![lookup_tool()])
                .with_max_tokens(4096),
        )
        .await?;
    let ChatOutcome::Success(first) = outcome else {
        bail!("Bedrock turn: {outcome:?}");
    };
    let tool_use_id = first
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .context("the Bedrock turn called no tool")?;
    let thinking_blocks = first
        .content
        .iter()
        .filter(|block| matches!(block, ContentBlock::Thinking { .. }))
        .count();
    ensure!(
        thinking_blocks > 0,
        "the Bedrock turn returned no thinking block to replay"
    );
    ensure!(
        first.content.iter().all(|block| match block {
            ContentBlock::Thinking { route, .. } | ContentBlock::RedactedThinking { route, .. } => {
                route.as_deref() == Some("bedrock")
            }
            _ => true,
        }),
        "Bedrock left a thinking block without its route, so clearing it tests nothing: {:?}",
        first.content
    );
    let unrouted = first
        .content
        .iter()
        .cloned()
        .map(|block| match block {
            ContentBlock::Thinking {
                thinking,
                signature,
                ..
            } => ContentBlock::Thinking {
                thinking,
                signature,
                route: None,
            },
            ContentBlock::RedactedThinking { data, .. } => {
                ContentBlock::RedactedThinking { data, route: None }
            }
            other => other,
        })
        .collect();
    let continuation = vec![
        question,
        Message {
            role: Role::Assistant,
            content: Content::Blocks(unrouted),
        },
        Message::tool_result(tool_use_id, "2026-12-01", false),
    ];
    let direct = AnthropicProvider::new(key, live.model.clone()).with_thinking(thinking);

    let in_tool_round = direct
        .chat(
            ChatRequest::new(system, continuation.clone())
                .with_tools(vec![lookup_tool()])
                .with_max_tokens(4096),
        )
        .await?;
    eprintln!(
        "Bedrock thinking blocks: {thinking_blocks}; Anthropic API replay in the tool round: \
         {in_tool_round:?}"
    );
    let ChatOutcome::Success(answer) = in_tool_round else {
        bail!("the Anthropic API refused Bedrock's block in a tool round: {in_tool_round:?}");
    };

    let mut next_question = continuation;
    next_question.push(Message {
        role: Role::Assistant,
        content: Content::Blocks(answer.content),
    });
    next_question.push(Message::user("And invoice 4472? Think it through first."));
    let after_the_round = direct
        .chat(
            ChatRequest::new(system, next_question)
                .with_tools(vec![lookup_tool()])
                .with_max_tokens(4096),
        )
        .await?;
    eprintln!("Anthropic API replay after the round: {after_the_round:?}");
    if !matches!(after_the_round, ChatOutcome::Success(_)) {
        bail!("the Anthropic API refused Bedrock's block on a new question: {after_the_round:?}");
    }
    Ok(())
}
