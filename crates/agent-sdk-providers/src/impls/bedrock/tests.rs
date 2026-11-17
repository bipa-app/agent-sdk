use std::sync::Arc;

use agent_sdk_foundation::llm::{
    CacheConfig, CacheTtl, Content, ContentBlock, ContentSource, Effort, Message, ResponseFormat,
    Role,
};
use anyhow::Context as _;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use super::*;
use crate::streaming::StreamAccumulator;

const ONE_PIXEL_PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

fn tool(name: &str) -> agent_sdk_foundation::llm::Tool {
    agent_sdk_foundation::llm::Tool {
        name: name.to_owned(),
        description: "desc".to_owned(),
        input_schema: json!({ "type": "object", "properties": { "q": { "type": "string" } } }),
        display_name: name.to_owned(),
        tier: agent_sdk_foundation::ToolTier::Observe,
        concurrency: agent_sdk_foundation::ToolConcurrency::Exclusive,
        replay: agent_sdk_foundation::ToolReplay::Unsafe,
    }
}

fn api_key() -> BedrockAuth {
    BedrockAuth::ApiKey("bedrock-api-key".to_owned())
}

fn provider(server: &MockServer, model: &str) -> BedrockProvider {
    BedrockProvider::new(
        api_key(),
        "us-east-1",
        model,
        format!("us.anthropic.{model}"),
    )
    .with_endpoint(server.uri())
}

fn agent_request() -> ChatRequest {
    ChatRequest::new(
        "You read documents.",
        vec![
            Message::user("first question"),
            Message::assistant("first answer"),
            Message::user("second question"),
        ],
    )
    .with_tools(vec![tool("lookup"), tool("pay")])
    .with_max_tokens(2048)
}

fn attachment_request() -> ChatRequest {
    ChatRequest::new(
        "You read documents.",
        vec![Message {
            role: Role::User,
            content: Content::Blocks(vec![
                ContentBlock::Image {
                    source: ContentSource::new("image/png", ONE_PIXEL_PNG),
                },
                ContentBlock::Document {
                    source: ContentSource::new(
                        "application/pdf",
                        base64::engine::general_purpose::STANDARD.encode(b"%PDF-1.4\n%%EOF\n"),
                    ),
                },
                ContentBlock::Text {
                    text: "what is due?".to_owned(),
                },
            ]),
        }],
    )
    .with_max_tokens(1024)
}

fn closed_format(optional_fields: usize) -> ResponseFormat {
    let mut properties = serde_json::Map::new();
    properties.insert("kind".to_owned(), json!({ "type": "string" }));
    for index in 0..optional_fields {
        properties.insert(format!("extra_{index}"), json!({ "type": "string" }));
    }
    ResponseFormat::new(
        "reading",
        json!({
            "type": "object",
            "properties": properties,
            "required": ["kind"],
            "additionalProperties": false
        }),
    )
}

/// The field set a live Bedrock `InvokeModel` call returned for Claude Sonnet
/// 5.5, with synthetic values.
fn success_body() -> serde_json::Value {
    json!({
        "id": "msg_bdrk_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{ "type": "text", "text": "ok" }],
        "container": null,
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "stop_details": null,
        "usage": {
            "input_tokens": 12,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 11005,
            "cache_creation": { "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 100 },
            "output_tokens": 7,
            "output_tokens_details": { "thinking_tokens": 3 },
            "service_tier": "standard"
        }
    })
}

async fn mount(server: &MockServer, template: ResponseTemplate) {
    Mock::given(method("POST"))
        .respond_with(template)
        .mount(server)
        .await;
}

async fn recorded(server: &MockServer) -> anyhow::Result<Vec<Request>> {
    server
        .received_requests()
        .await
        .context("request recording is disabled")
}

async fn bedrock_body(
    model: &str,
    thinking: Option<ThinkingConfig>,
    request: ChatRequest,
) -> anyhow::Result<serde_json::Value> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(200).set_body_json(success_body()),
    )
    .await;
    let provider = provider(&server, model);
    let provider = match thinking {
        Some(thinking) => provider.with_thinking(thinking),
        None => provider,
    };
    provider.chat(request).await?;
    let requests = recorded(&server).await?;
    anyhow::ensure!(
        requests.len() == 1,
        "expected one request, got {}",
        requests.len()
    );
    Ok(serde_json::from_slice(&requests[0].body)?)
}

async fn anthropic_body(
    model: &str,
    thinking: Option<ThinkingConfig>,
    request: ChatRequest,
) -> anyhow::Result<serde_json::Value> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(200).set_body_json(success_body()),
    )
    .await;
    let provider = AnthropicProvider::new("test-key", model).with_base_url(server.uri());
    let provider = match thinking {
        Some(thinking) => provider.with_thinking(thinking),
        None => provider,
    };
    provider.chat(request).await?;
    let requests = recorded(&server).await?;
    anyhow::ensure!(
        requests.len() == 1,
        "expected one request, got {}",
        requests.len()
    );
    Ok(serde_json::from_slice(&requests[0].body)?)
}

/// The Bedrock body is the Anthropic body with the model and the stream flag
/// moved off it and the Bedrock API version on it. Returns the Bedrock betas.
async fn assert_body_parity(
    model: &str,
    thinking: Option<ThinkingConfig>,
    request: ChatRequest,
) -> anyhow::Result<serde_json::Value> {
    let mut bedrock = bedrock_body(model, thinking.clone(), request.clone()).await?;
    let mut anthropic = anthropic_body(model, thinking, request).await?;
    let bedrock_fields = bedrock.as_object_mut().context("bedrock body")?;
    assert_eq!(
        bedrock_fields.remove("anthropic_version"),
        Some(json!("bedrock-2023-05-31"))
    );
    assert!(!bedrock_fields.contains_key("model"));
    assert!(!bedrock_fields.contains_key("stream"));
    let betas = bedrock_fields.remove("anthropic_beta");
    let anthropic_fields = anthropic.as_object_mut().context("anthropic body")?;
    assert_eq!(anthropic_fields.remove("model"), Some(json!(model)));
    assert_eq!(anthropic_fields.remove("stream"), Some(json!(false)));
    assert_eq!(bedrock, anthropic);
    Ok(betas.unwrap_or(serde_json::Value::Null))
}

#[test]
fn inference_profile_ids_follow_the_model_cards() {
    assert_eq!(
        bedrock_inference_profile_id(MODEL_SONNET_55, BedrockInferenceProfile::Us).as_deref(),
        Some("us.anthropic.claude-sonnet-5-5")
    );
    assert_eq!(
        bedrock_inference_profile_id(MODEL_OPUS_55, BedrockInferenceProfile::Global).as_deref(),
        Some("global.anthropic.claude-opus-5-5")
    );
    assert_eq!(
        bedrock_inference_profile_id(MODEL_HAIKU_45, BedrockInferenceProfile::Us).as_deref(),
        Some("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    );
    // An undated alias names its snapshot; another family or another dated
    // snapshot does not.
    assert_eq!(
        bedrock_inference_profile_id("claude-sonnet-4-5", BedrockInferenceProfile::Us).as_deref(),
        Some("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    );
    assert_eq!(bedrock_base_model_id("claude-sonnet-4"), None);
    assert_eq!(bedrock_base_model_id("claude-sonnet-4-5-20990101"), None);
    // Undocumented pairs and models have no id.
    assert_eq!(
        bedrock_inference_profile_id(MODEL_SONNET_55, BedrockInferenceProfile::Jp),
        None
    );
    assert_eq!(bedrock_base_model_id("claude-sonnet-4-20250514"), None);
    assert_eq!(bedrock_base_model_id("claude-3-5-haiku-20241022"), None);
}

#[test]
fn every_bedrock_model_has_an_anthropic_capability_row() {
    for model in BEDROCK_MODELS {
        assert!(
            crate::model_capabilities::get_model_capabilities("anthropic", model.anthropic_id)
                .is_some(),
            "{} has no capability row",
            model.anthropic_id
        );
    }
}

#[test]
fn identity_is_the_anthropic_model_on_the_bedrock_route() -> anyhow::Result<()> {
    let provider = BedrockProvider::with_inference_profile(
        api_key(),
        "us-east-1",
        MODEL_SONNET_55,
        BedrockInferenceProfile::Us,
    )?;
    assert_eq!(provider.model(), "claude-sonnet-5-5");
    assert_eq!(provider.provider(), "anthropic");
    assert_eq!(provider.route(), "bedrock");
    assert_eq!(provider.model_id(), "us.anthropic.claude-sonnet-5-5");
    assert!(provider.capabilities().is_some());
    assert!(
        BedrockProvider::with_inference_profile(
            api_key(),
            "us-east-1",
            MODEL_SONNET_55,
            BedrockInferenceProfile::Jp,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn the_model_id_is_one_encoded_path_segment() {
    let provider = BedrockProvider::new(
        api_key(),
        "us-east-1",
        "claude-opus-4-5-20251101",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-5-20251101-v1:0",
    );
    assert_eq!(
        provider.invoke_url(true),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fus.anthropic.claude-opus-4-5-20251101-v1%3A0/invoke-with-response-stream"
    );
}

#[tokio::test]
async fn cache_breakpoints_match_the_anthropic_body() -> anyhow::Result<()> {
    let betas = assert_body_parity(MODEL_SONNET_55, None, agent_request()).await?;
    assert_eq!(betas, serde_json::Value::Null);
    let body = bedrock_body(MODEL_SONNET_55, None, agent_request()).await?;
    assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
    let last = body["messages"][2]["content"]
        .as_array()
        .and_then(|blocks| blocks.last())
        .context("last user block")?;
    assert_eq!(last["cache_control"]["type"], "ephemeral");
    Ok(())
}

#[tokio::test]
async fn a_one_hour_cache_ttl_matches_the_anthropic_body() -> anyhow::Result<()> {
    let request = agent_request().with_cache(CacheConfig::default().with_ttl(CacheTtl::OneHour));
    assert_body_parity(MODEL_SONNET_55, None, request.clone()).await?;
    let body = bedrock_body(MODEL_SONNET_55, None, request).await?;
    assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
    Ok(())
}

#[tokio::test]
async fn adaptive_thinking_and_effort_match_the_anthropic_body() -> anyhow::Result<()> {
    let thinking = ThinkingConfig::adaptive_with_effort(Effort::Low);
    let betas =
        assert_body_parity(MODEL_SONNET_55, Some(thinking.clone()), agent_request()).await?;
    assert_eq!(betas, serde_json::Value::Null);
    let body = bedrock_body(MODEL_SONNET_55, Some(thinking), agent_request()).await?;
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"]["effort"], "low");
    Ok(())
}

#[tokio::test]
async fn budget_thinking_models_carry_the_interleaved_beta_in_the_body() -> anyhow::Result<()> {
    let betas = assert_body_parity(
        MODEL_HAIKU_45,
        Some(ThinkingConfig::new(2048)),
        agent_request(),
    )
    .await?;
    assert_eq!(betas, json!(["interleaved-thinking-2025-05-14"]));
    Ok(())
}

#[tokio::test]
async fn image_and_pdf_attachments_match_the_anthropic_body() -> anyhow::Result<()> {
    assert_body_parity(MODEL_SONNET_55, None, attachment_request()).await?;
    let body = bedrock_body(MODEL_SONNET_55, None, attachment_request()).await?;
    let blocks = body["messages"][0]["content"]
        .as_array()
        .context("content blocks")?;
    assert_eq!(blocks[0]["type"], "image");
    assert_eq!(blocks[0]["source"]["type"], "base64");
    assert_eq!(blocks[1]["type"], "document");
    assert_eq!(blocks[1]["source"]["media_type"], "application/pdf");
    Ok(())
}

#[tokio::test]
async fn structured_output_matches_the_anthropic_body() -> anyhow::Result<()> {
    let request = ChatRequest::new("Read it.", vec![Message::user("a receipt")])
        .with_response_format(closed_format(3));
    assert_body_parity(MODEL_SONNET_55, None, request.clone()).await?;
    let body = bedrock_body(MODEL_SONNET_55, None, request).await?;
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    Ok(())
}

#[tokio::test]
async fn a_forced_tool_matches_the_anthropic_body_where_bedrock_allows_it() -> anyhow::Result<()> {
    let request = agent_request().with_tool_choice(ToolChoice::Tool("pay".to_owned()));
    assert_body_parity(
        MODEL_HAIKU_55,
        Some(ThinkingConfig::adaptive()),
        request.clone(),
    )
    .await?;
    let body = bedrock_body(MODEL_HAIKU_55, Some(ThinkingConfig::adaptive()), request).await?;
    assert_eq!(
        body["tool_choice"],
        json!({ "type": "tool", "name": "pay" })
    );
    assert!(body.get("thinking").is_none());
    Ok(())
}

#[tokio::test]
async fn a_forced_tool_on_sonnet_55_is_refused_before_the_wire() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .expect(0)
        .mount(&server)
        .await;
    let request = agent_request().with_tool_choice(ToolChoice::Tool("pay".to_owned()));
    let outcome = provider(&server, MODEL_SONNET_55).chat(request).await?;
    let ChatOutcome::InvalidRequest(message) = outcome else {
        anyhow::bail!("expected a refusal, got {outcome:?}");
    };
    assert!(message.contains("auto or none"), "{message}");
    Ok(())
}

#[tokio::test]
async fn an_undated_alias_takes_its_snapshots_card() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(200).set_body_json(success_body()),
    )
    .await;
    let provider = BedrockProvider::new(
        api_key(),
        "us-east-1",
        "claude-sonnet-4-5",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .with_endpoint(server.uri());
    assert!(matches!(
        provider.structured_output_support_for(&closed_format(3)),
        StructuredOutputSupport::Native
    ));
    provider
        .chat(
            ChatRequest::new("Read it.", vec![Message::user("a receipt")])
                .with_response_format(closed_format(3)),
        )
        .await?;
    let requests = recorded(&server).await?;
    let body: serde_json::Value =
        serde_json::from_slice(&requests.first().context("no request")?.body)?;
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    Ok(())
}

#[test]
fn a_schema_past_bedrocks_optional_limit_falls_back_to_tool_forcing() {
    let provider = BedrockProvider::new(api_key(), "us-east-1", MODEL_HAIKU_55, "id");
    assert!(matches!(
        provider.structured_output_support_for(&closed_format(12)),
        StructuredOutputSupport::Native
    ));
    assert!(matches!(
        provider.structured_output_support_for(&closed_format(13)),
        StructuredOutputSupport::ToolForcing
    ));
    let unknown = BedrockProvider::new(api_key(), "us-east-1", "claude-3-5-haiku-20241022", "id");
    assert!(matches!(
        unknown.structured_output_support(),
        StructuredOutputSupport::ToolForcing
    ));
}

#[tokio::test]
async fn a_direct_request_with_a_schema_bedrock_cannot_take_is_refused() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .expect(0)
        .mount(&server)
        .await;
    let request = ChatRequest::new("Read it.", vec![Message::user("x")])
        .with_response_format(closed_format(13));
    let outcome = provider(&server, MODEL_SONNET_55).chat(request).await?;
    assert!(
        matches!(&outcome, ChatOutcome::InvalidRequest(message) if message.contains("12")),
        "{outcome:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_forced_tool_carries_a_schema_bedrock_cannot_take() -> anyhow::Result<()> {
    let request = ChatRequest::new("Read it.", vec![Message::user("x")])
        .with_tools(vec![tool("respond")])
        .with_tool_choice(ToolChoice::Tool("respond".to_owned()))
        .with_response_format(closed_format(13));
    let body = bedrock_body(MODEL_HAIKU_55, None, request).await?;
    assert!(body.get("output_config").is_none(), "{body}");
    assert_eq!(body["tool_choice"]["name"], "respond");
    Ok(())
}

#[test]
fn optional_properties_are_counted_across_nested_objects() {
    let schema = json!({
        "type": "object",
        "properties": {
            "a": { "type": "string" },
            "b": {
                "type": "object",
                "properties": { "c": { "type": "string" }, "d": { "type": "string" } },
                "required": ["c"]
            },
            "list": { "type": "array", "items": { "$ref": "#/$defs/item" } },
            "choice": { "anyOf": [
                { "type": "object", "properties": { "e": { "type": "string" } } },
                { "type": "null" }
            ] }
        },
        "required": ["a", "b", "list", "choice"],
        "$defs": { "item": { "type": "object", "properties": { "f": {}, "g": {} }, "required": ["f"] } }
    });
    // d, e and g are optional.
    assert_eq!(optional_property_count(&schema), 3);
}

#[test]
fn budget_thinking_is_refused_on_adaptive_models() {
    let provider = BedrockProvider::new(api_key(), "us-east-1", MODEL_SONNET_55, "id");
    assert!(
        provider
            .validate_thinking_config(Some(&ThinkingConfig::new(1024)))
            .is_err()
    );
    assert!(
        provider
            .validate_thinking_config(Some(&ThinkingConfig::adaptive()))
            .is_ok()
    );
}

#[test]
fn http_rejections_map_to_retry_classes() {
    let classify = |status: u16, error_type: Option<&str>| {
        classify(
            StatusCode::from_u16(status).unwrap_or(StatusCode::IM_A_TEAPOT),
            error_type,
            Some(Duration::from_secs(7)),
            r#"{"message":"m"}"#,
        )
    };
    assert_eq!(
        classify(429, Some("ThrottlingException")),
        Rejection::RateLimited(Some(Duration::from_secs(7)))
    );
    assert_eq!(
        classify(400, Some("ServiceQuotaExceededException")),
        Rejection::RateLimited(Some(Duration::from_secs(7)))
    );
    assert_eq!(
        classify(529, None),
        Rejection::RateLimited(Some(Duration::from_secs(7)))
    );
    for status in [408, 424, 500, 503] {
        assert!(
            matches!(classify(status, None), Rejection::ServerError(_)),
            "{status}"
        );
    }
    assert!(matches!(
        classify(400, Some("ValidationException")),
        Rejection::InvalidRequest(_)
    ));
    assert!(matches!(
        classify(403, Some("AccessDeniedException")),
        Rejection::InvalidRequest(_)
    ));
    assert!(matches!(
        classify(403, Some("ExpiredTokenException")),
        Rejection::ExpiredCredentials(_)
    ));
}

#[tokio::test]
async fn a_429_is_rate_limited_with_its_retry_after() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(429)
            .insert_header(
                "x-amzn-ErrorType",
                "ThrottlingException:http://internal.amazon.com/coral/com.amazon.bedrock/",
            )
            .insert_header("Retry-After", "3")
            .set_body_json(json!({ "message": "Too many requests" })),
    )
    .await;
    let outcome = provider(&server, MODEL_SONNET_55)
        .chat(agent_request())
        .await?;
    assert!(
        matches!(outcome, ChatOutcome::RateLimited(Some(delay)) if delay == Duration::from_secs(3)),
        "{outcome:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_prompt_too_long_rejection_keeps_its_message() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(400)
            .insert_header(
                "x-amzn-ErrorType",
                "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/",
            )
            .set_body_json(
                json!({ "message": "prompt is too long: 1010596 tokens > 1000000 maximum" }),
            ),
    )
    .await;
    let outcome = provider(&server, MODEL_SONNET_55)
        .chat(agent_request())
        .await?;
    let ChatOutcome::InvalidRequest(message) = outcome else {
        anyhow::bail!("expected InvalidRequest, got {outcome:?}");
    };
    assert!(message.contains("prompt is too long"), "{message}");
    assert!(message.contains("ValidationException"), "{message}");
    Ok(())
}

#[tokio::test]
async fn a_success_maps_usage_with_cache_tokens() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(200).set_body_json(success_body()),
    )
    .await;
    let outcome = provider(&server, MODEL_SONNET_55)
        .chat(agent_request())
        .await?;
    let ChatOutcome::Success(response) = outcome else {
        anyhow::bail!("expected success, got {outcome:?}");
    };
    assert_eq!(response.model, "claude-sonnet-5-5");
    assert_eq!(response.usage.input_tokens, 12 + 100 + 11_005);
    assert_eq!(response.usage.cached_input_tokens, 11_005);
    assert_eq!(response.usage.cache_creation_input_tokens, 100);
    assert_eq!(response.usage.output_tokens, 7);
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    Ok(())
}

#[tokio::test]
async fn the_response_names_the_anthropic_model() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    let mut body = success_body();
    body["model"] = json!("anthropic.claude-sonnet-5-5");
    mount(&server, ResponseTemplate::new(200).set_body_json(body)).await;
    let outcome = provider(&server, MODEL_SONNET_55)
        .chat(agent_request())
        .await?;
    let ChatOutcome::Success(response) = outcome else {
        anyhow::bail!("expected success, got {outcome:?}");
    };
    assert_eq!(response.model, "claude-sonnet-5-5");
    Ok(())
}

#[tokio::test]
async fn an_api_key_rides_the_bearer_header() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(200).set_body_json(success_body()),
    )
    .await;
    provider(&server, MODEL_SONNET_55)
        .chat(agent_request())
        .await?;
    let requests = recorded(&server).await?;
    let request = requests.first().context("no request")?;
    assert_eq!(
        request.url.path(),
        "/model/us.anthropic.claude-sonnet-5-5/invoke"
    );
    assert_eq!(
        request
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer bedrock-api-key")
    );
    assert_eq!(
        request.headers.get("accept").and_then(|v| v.to_str().ok()),
        Some("application/json")
    );
    Ok(())
}

/// A credentials source that always fails, permanently or not.
#[derive(Debug)]
struct FailingCredentials {
    refused: bool,
}

impl ProvideCredentials for FailingCredentials {
    fn provide_credentials<'a>(
        &'a self,
    ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        aws_credential_types::provider::future::ProvideCredentials::ready(Err(if self.refused {
            CredentialsError::invalid_configuration("STS answered 403: AccessDenied")
        } else {
            CredentialsError::provider_error("connection reset by the STS endpoint")
        }))
    }
}

fn expired_token_response() -> ResponseTemplate {
    ResponseTemplate::new(403)
        .insert_header("x-amzn-ErrorType", "ExpiredTokenException")
        .set_body_json(
            json!({ "message": "The security token included in the request is expired" }),
        )
}

/// A mock STS that issues temporary credentials for an hour, and web identity
/// credentials that read their token from a file and ask that STS.
async fn web_identity() -> anyhow::Result<(
    MockServer,
    tempfile::NamedTempFile,
    Arc<WebIdentityCredentials>,
)> {
    let token_file = tempfile::NamedTempFile::new()?;
    std::fs::write(token_file.path(), "file.token")?;
    let sts = MockServer::start().await;
    let expiration =
        aws_smithy_types::DateTime::from(std::time::SystemTime::now() + Duration::from_hours(1))
            .fmt(aws_smithy_types::date_time::Format::DateTime)?;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(format!(
            "<AssumeRoleWithWebIdentityResponse><AssumeRoleWithWebIdentityResult><Credentials>\
             <SessionToken>session-token-example</SessionToken>\
             <SecretAccessKey>wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY</SecretAccessKey>\
             <Expiration>{expiration}</Expiration><AccessKeyId>AKIDEXAMPLE</AccessKeyId>\
             </Credentials></AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>"
        )))
        .mount(&sts)
        .await;
    let credentials = Arc::new(
        WebIdentityCredentials::new(
            "arn:aws:iam::123456789012:role/example",
            "session",
            WebIdentityToken::File(token_file.path().to_path_buf()),
            "us-east-1",
        )
        .with_sts_endpoint(sts.uri()),
    );
    Ok((sts, token_file, credentials))
}

fn web_identity_provider(
    server: &MockServer,
    credentials: Arc<WebIdentityCredentials>,
) -> BedrockProvider {
    BedrockProvider::new(
        BedrockAuth::WebIdentity(credentials),
        "us-east-1",
        MODEL_SONNET_55,
        "us.anthropic.claude-sonnet-5-5",
    )
    .with_endpoint(server.uri())
}

#[tokio::test]
async fn sigv4_requests_are_signed_for_bedrock_and_retry_once_on_expired_credentials()
-> anyhow::Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(expired_token_response())
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount(
        &server,
        ResponseTemplate::new(200).set_body_json(success_body()),
    )
    .await;
    let (sts, _token, credentials) = web_identity().await?;

    let outcome = web_identity_provider(&server, credentials)
        .chat(agent_request())
        .await?;
    assert!(matches!(outcome, ChatOutcome::Success(_)), "{outcome:?}");
    // The expired credentials were dropped and fetched again.
    assert_eq!(recorded(&sts).await?.len(), 2);

    let requests = recorded(&server).await?;
    assert_eq!(requests.len(), 2);
    let request = requests.last().context("no request")?;
    let authorization = request
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .context("no authorization")?;
    assert!(
        authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
            && authorization.contains("/us-east-1/bedrock/aws4_request")
            && authorization.contains("SignedHeaders=host;x-amz-date;x-amz-security-token"),
        "{authorization}"
    );
    assert_eq!(
        request
            .headers
            .get("x-amz-security-token")
            .and_then(|v| v.to_str().ok()),
        Some("session-token-example")
    );
    Ok(())
}

#[tokio::test]
async fn a_streamed_request_retries_once_on_expired_credentials() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(expired_token_response())
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(
            "/model/us.anthropic.claude-sonnet-5-5/invoke-with-response-stream",
        ))
        .respond_with(event_stream_response(frames(&tool_call_events())?))
        .mount(&server)
        .await;
    let (sts, _token, credentials) = web_identity().await?;

    let provider = web_identity_provider(&server, credentials);
    let mut stream = provider.chat_stream(agent_request());
    let mut deltas = Vec::new();
    while let Some(delta) = stream.next().await {
        deltas.push(delta?);
    }
    assert!(
        matches!(
            deltas.last(),
            Some(StreamDelta::Done {
                stop_reason: Some(StopReason::ToolUse),
                ..
            })
        ),
        "{deltas:?}"
    );
    assert_eq!(recorded(&sts).await?.len(), 2);
    assert_eq!(recorded(&server).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn a_second_expired_token_is_an_invalid_request_after_one_retry() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(&server, expired_token_response()).await;
    let (sts, _token, credentials) = web_identity().await?;
    let outcome = web_identity_provider(&server, credentials)
        .chat(agent_request())
        .await?;
    assert!(
        matches!(outcome, ChatOutcome::InvalidRequest(_)),
        "{outcome:?}"
    );
    assert_eq!(recorded(&sts).await?.len(), 2);
    assert_eq!(recorded(&server).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn static_credentials_do_not_retry_an_expired_token() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(&server, expired_token_response()).await;
    let provider = BedrockProvider::new(
        BedrockAuth::Credentials(SharedCredentialsProvider::new(Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("expired".to_owned()),
            None,
            "test",
        ))),
        "us-east-1",
        MODEL_SONNET_55,
        "us.anthropic.claude-sonnet-5-5",
    )
    .with_endpoint(server.uri());
    let outcome = provider.chat(agent_request()).await?;
    assert!(
        matches!(outcome, ChatOutcome::InvalidRequest(_)),
        "{outcome:?}"
    );
    assert_eq!(recorded(&server).await?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn refused_credentials_are_an_invalid_request_and_others_are_retried() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    let provider = |refused| {
        BedrockProvider::new(
            BedrockAuth::Credentials(SharedCredentialsProvider::new(FailingCredentials {
                refused,
            })),
            "us-east-1",
            MODEL_SONNET_55,
            "us.anthropic.claude-sonnet-5-5",
        )
        .with_endpoint(server.uri())
    };

    let outcome = provider(true).chat(agent_request()).await?;
    assert!(
        matches!(&outcome, ChatOutcome::InvalidRequest(message) if message.contains("AccessDenied")),
        "{outcome:?}"
    );
    assert!(provider(false).chat(agent_request()).await.is_err());

    for (refused, expected) in [
        (true, StreamErrorKind::InvalidRequest),
        (false, StreamErrorKind::ServerError),
    ] {
        let failing = provider(refused);
        let mut stream = failing.chat_stream(agent_request());
        let delta = stream.next().await.context("no delta")??;
        assert!(
            matches!(delta, StreamDelta::Error { kind, .. } if kind == expected),
            "{delta:?}"
        );
    }
    assert!(recorded(&server).await?.is_empty());
    Ok(())
}

// ── Streaming ────────────────────────────────────────────────────────────

fn chunk(event: &serde_json::Value) -> anyhow::Result<Vec<u8>> {
    let bytes = base64::engine::general_purpose::STANDARD.encode(event.to_string());
    event_stream::tests::encode(
        &[
            (":event-type", "chunk"),
            (":content-type", "application/json"),
            (":message-type", "event"),
        ],
        json!({ "bytes": bytes }).to_string().as_bytes(),
    )
}

fn exception(exception_type: &str, payload: &serde_json::Value) -> anyhow::Result<Vec<u8>> {
    event_stream::tests::encode(
        &[
            (":exception-type", exception_type),
            (":content-type", "application/json"),
            (":message-type", "exception"),
        ],
        payload.to_string().as_bytes(),
    )
}

/// The frames of `events`, one `chunk` each.
fn frames(events: &[serde_json::Value]) -> anyhow::Result<Vec<u8>> {
    Ok(events
        .iter()
        .map(chunk)
        .collect::<anyhow::Result<Vec<_>>>()?
        .concat())
}

fn tool_call_events() -> Vec<serde_json::Value> {
    vec![
        json!({ "type": "message_start", "message": { "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-sonnet-5-5", "content": [], "stop_reason": null,
            "usage": { "input_tokens": 20, "cache_creation_input_tokens": 300, "cache_read_input_tokens": 4000, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "ping" }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "Checking." } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "input_json_delta", "partial_json": "{\"q\":\"due\"}" } }),
        json!({ "type": "content_block_stop", "index": 1 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 42 } }),
        json!({ "type": "message_stop", "amazon-bedrock-invocationMetrics": { "inputTokenCount": 20, "outputTokenCount": 42, "invocationLatency": 900, "firstByteLatency": 300 } }),
    ]
}

async fn stream_deltas(server: &MockServer, model: &str) -> anyhow::Result<Vec<StreamDelta>> {
    let provider = provider(server, model);
    let mut stream = provider.chat_stream(agent_request());
    let mut deltas = Vec::new();
    while let Some(delta) = stream.next().await {
        deltas.push(delta?);
    }
    Ok(deltas)
}

fn event_stream_response(body: Vec<u8>) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "application/vnd.amazon.eventstream")
        .insert_header("x-amzn-bedrock-content-type", "application/json")
        .set_body_bytes(body)
}

#[tokio::test]
async fn a_streamed_tool_call_decodes_with_usage_and_route() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/model/us.anthropic.claude-sonnet-5-5/invoke-with-response-stream",
        ))
        .respond_with(event_stream_response(frames(&tool_call_events())?))
        .mount(&server)
        .await;
    let deltas = stream_deltas(&server, MODEL_SONNET_55).await?;

    let requests = recorded(&server).await?;
    assert_eq!(
        requests
            .first()
            .and_then(|r| r.headers.get("accept"))
            .and_then(|v| v.to_str().ok()),
        Some("application/vnd.amazon.eventstream")
    );
    assert!(deltas.iter().any(|d| matches!(d, StreamDelta::KeepAlive)));
    assert!(
        deltas
            .iter()
            .any(|d| matches!(d, StreamDelta::TextDelta { delta, .. } if delta == "Checking."))
    );
    assert!(deltas.iter().any(|d| matches!(
        d,
        StreamDelta::ToolUseStart { id, name, .. } if id == "toolu_1" && name == "lookup"
    )));
    let usage = deltas
        .iter()
        .find_map(|d| match d {
            StreamDelta::Usage(usage) => Some(usage.clone()),
            _ => None,
        })
        .context("no usage")?;
    assert_eq!(usage.input_tokens, 20 + 300 + 4000);
    assert_eq!(usage.cached_input_tokens, 4000);
    assert_eq!(usage.cache_creation_input_tokens, 300);
    assert_eq!(usage.output_tokens, 42);
    let Some(StreamDelta::Done {
        stop_reason,
        served_route,
        served_model,
    }) = deltas.last()
    else {
        anyhow::bail!("stream did not end with Done: {deltas:?}");
    };
    assert_eq!(*stop_reason, Some(StopReason::ToolUse));
    assert_eq!(served_route.as_deref(), Some("bedrock"));
    assert_eq!(served_model.as_deref(), Some("claude-sonnet-5-5"));

    let mut accumulator = StreamAccumulator::new();
    for delta in &deltas {
        accumulator.apply(delta);
    }
    let blocks = accumulator.into_content_blocks();
    assert!(blocks.iter().any(|block| matches!(
        block,
        ContentBlock::ToolUse { input, .. } if input == &json!({ "q": "due" })
    )));
    Ok(())
}

#[tokio::test]
async fn the_stream_ends_at_message_stop() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    let mut body = frames(&tool_call_events())?;
    body.extend(chunk(&json!({
        "type": "content_block_delta", "index": 0,
        "delta": { "type": "text_delta", "text": "after the stop" }
    }))?);
    mount(&server, event_stream_response(body)).await;
    let deltas = stream_deltas(&server, MODEL_SONNET_55).await?;
    assert!(
        matches!(deltas.last(), Some(StreamDelta::Done { .. })),
        "{deltas:?}"
    );
    assert!(
        !deltas.iter().any(
            |d| matches!(d, StreamDelta::TextDelta { delta, .. } if delta == "after the stop")
        )
    );
    Ok(())
}

#[tokio::test]
async fn stream_exceptions_map_to_retry_classes() -> anyhow::Result<()> {
    let cases = [
        (
            "throttlingException",
            json!({ "message": "slow down" }),
            StreamErrorKind::RateLimited(None),
        ),
        (
            "serviceUnavailableException",
            json!({ "message": "busy" }),
            StreamErrorKind::ServerError,
        ),
        (
            "modelTimeoutException",
            json!({ "message": "late" }),
            StreamErrorKind::ServerError,
        ),
        (
            "internalServerException",
            json!({ "message": "oops" }),
            StreamErrorKind::ServerError,
        ),
        (
            "validationException",
            json!({ "message": "bad" }),
            StreamErrorKind::InvalidRequest,
        ),
        (
            "modelStreamErrorException",
            json!({ "message": "overloaded", "originalStatusCode": 529 }),
            StreamErrorKind::RateLimited(None),
        ),
        (
            "modelStreamErrorException",
            json!({ "message": "bad input", "originalStatusCode": 400 }),
            StreamErrorKind::InvalidRequest,
        ),
        (
            "modelStreamErrorException",
            json!({ "message": "failed" }),
            StreamErrorKind::ServerError,
        ),
        (
            "someNewException",
            json!({ "message": "new" }),
            StreamErrorKind::ServerError,
        ),
    ];
    for (exception_type, payload, expected) in cases {
        let server = MockServer::start().await;
        let mut body = chunk(&tool_call_events()[0])?;
        body.extend(exception(exception_type, &payload)?);
        mount(&server, event_stream_response(body)).await;
        let deltas = stream_deltas(&server, MODEL_SONNET_55).await?;
        let Some(StreamDelta::Error { message, kind }) = deltas.last() else {
            anyhow::bail!("{exception_type}: no error in {deltas:?}");
        };
        assert_eq!(*kind, expected, "{exception_type}");
        assert!(message.contains(exception_type), "{message}");
        assert_eq!(
            deltas
                .iter()
                .filter(|d| matches!(d, StreamDelta::Error { .. }))
                .count(),
            1,
            "{exception_type}: {deltas:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_stream_cut_before_message_stop_is_a_server_error() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    let events = tool_call_events();
    let mut body = frames(&events[..events.len() - 1])?;
    // Half of a frame is left buffered when the body ends.
    let tail = chunk(&events[events.len() - 1])?;
    body.extend_from_slice(&tail[..tail.len() / 2]);
    mount(&server, event_stream_response(body)).await;
    let deltas = stream_deltas(&server, MODEL_SONNET_55).await?;
    assert!(
        matches!(
            deltas.last(),
            Some(StreamDelta::Error {
                kind: StreamErrorKind::ServerError,
                ..
            })
        ),
        "{deltas:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_corrupt_frame_ends_the_stream_with_a_server_error() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    let mut body = frames(&tool_call_events())?;
    let last = body.len() - 1;
    body[last] ^= 0xff;
    mount(&server, event_stream_response(body)).await;
    let deltas = stream_deltas(&server, MODEL_SONNET_55).await?;
    let Some(StreamDelta::Error { message, kind }) = deltas.last() else {
        anyhow::bail!("no error in {deltas:?}");
    };
    assert_eq!(*kind, StreamErrorKind::ServerError);
    assert!(message.contains("event-stream frame"), "{message}");
    Ok(())
}

#[tokio::test]
async fn an_http_throttle_before_the_stream_is_rate_limited() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(429)
            .insert_header("x-amzn-ErrorType", "ThrottlingException")
            .set_body_json(json!({ "message": "Too many tokens" })),
    )
    .await;
    let deltas = stream_deltas(&server, MODEL_SONNET_55).await?;
    assert!(
        matches!(
            deltas.as_slice(),
            [StreamDelta::Error {
                kind: StreamErrorKind::RateLimited(None),
                ..
            }]
        ),
        "{deltas:?}"
    );
    Ok(())
}

#[tokio::test]
async fn slow_headers_end_the_stream_as_a_lost_connection() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(200).set_delay(Duration::from_secs(2)),
    )
    .await;
    let provider = provider(&server, MODEL_SONNET_55)
        .with_stream_stall_timeouts(Duration::from_millis(200), Duration::from_millis(200));
    let mut stream = provider.chat_stream(agent_request());
    let delta = stream.next().await.context("no delta")??;
    assert!(
        matches!(
            delta,
            StreamDelta::Error {
                kind: StreamErrorKind::ConnectionLost,
                ..
            }
        ),
        "{delta:?}"
    );
    Ok(())
}
