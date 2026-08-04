#![cfg(feature = "openai")]

use std::num::NonZeroU32;
use std::time::Duration;

use agent_sdk_providers::{
    ChatOutcome, ChatRequest, EmbeddingError, EmbeddingRequest, EmbeddingResponse, LlmProvider,
    MAX_EMBEDDING_BATCH_SIZE, MAX_EMBEDDING_DIMENSIONS, MAX_EMBEDDING_INPUT_BYTES,
    MAX_EMBEDDING_RESPONSE_BYTES, MAX_EMBEDDING_TOTAL_INPUT_BYTES, OpenAIProvider,
};
#[cfg(feature = "gemini")]
use agent_sdk_providers::GeminiProvider;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request(inputs: Vec<&str>) -> EmbeddingRequest {
    EmbeddingRequest::new(
        "text-embedding-3-large",
        inputs.into_iter().map(str::to_owned).collect(),
    )
}

async fn call_with_response(
    response: ResponseTemplate,
    request: EmbeddingRequest,
) -> Result<std::result::Result<EmbeddingResponse, EmbeddingError>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(response)
        .mount(&server)
        .await;
    let provider = OpenAIProvider::with_base_url(
        "test-key",
        "chat-model-is-not-used-for-embeddings",
        format!("{}/v1", server.uri()),
    );
    Ok(provider.embed(request).await)
}

fn require_invalid_request(
    result: std::result::Result<EmbeddingResponse, EmbeddingError>,
) -> Result<()> {
    match result {
        Err(EmbeddingError::InvalidRequest { .. }) => Ok(()),
        Ok(_) => bail!("invalid embedding request unexpectedly succeeded"),
        Err(error) => bail!("expected invalid-request error, got {error:?}"),
    }
}

fn require_invalid_response(
    result: std::result::Result<EmbeddingResponse, EmbeddingError>,
) -> Result<()> {
    match result {
        Err(EmbeddingError::InvalidResponse { .. }) => Ok(()),
        Ok(_) => bail!("invalid embedding response unexpectedly succeeded"),
        Err(error) => bail!("expected invalid-response error, got {error:?}"),
    }
}

#[tokio::test]
async fn openai_embedding_request_reuses_base_url_auth_headers_and_restores_input_order() -> Result<()>
{
    let server = MockServer::start().await;
    let dimensions = NonZeroU32::new(3).context("three is nonzero")?;
    Mock::given(method("POST"))
        .and(path("/custom/v1/embeddings"))
        .and(header("authorization", "Bearer sdk-secret"))
        .and(header("x-gateway-auth", "gateway-secret"))
        .and(body_json(json!({
            "model": "text-embedding-3-large",
            "input": ["first", "second"],
            "dimensions": 3,
            "encoding_format": "float"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [4.0, 5.0, 6.0]},
                {"object": "embedding", "index": 0, "embedding": [1.0, 2.0, 3.0]}
            ],
            "model": "text-embedding-3-large",
            "usage": {"prompt_tokens": 7, "total_tokens": 7}
        })))
        .mount(&server)
        .await;

    let provider = OpenAIProvider::with_base_url(
        "sdk-secret",
        "unrelated-chat-model",
        format!("{}/custom/v1", server.uri()),
    )
    .with_extra_headers(vec![(
        "x-gateway-auth".to_owned(),
        "gateway-secret".to_owned(),
    )]);
    let response = provider
        .embed(request(vec!["first", "second"]).with_dimensions(dimensions))
        .await
        .context("embedding request succeeds")?;

    assert_eq!(
        response.vectors,
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
    );
    Ok(())
}

#[tokio::test]
async fn optional_dimensions_are_omitted_from_the_wire_request() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_json(json!({
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "encoding_format": "float"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [0.25, 0.5]}],
            "model": "text-embedding-3-small"
        })))
        .mount(&server)
        .await;
    let provider = OpenAIProvider::with_base_url("key", "chat-model", server.uri());

    let response = provider
        .embed(EmbeddingRequest::new(
            "text-embedding-3-small",
            vec!["hello".to_owned()],
        ))
        .await
        .context("embedding request succeeds without dimensions")?;

    assert_eq!(response.vectors, vec![vec![0.25, 0.5]]);
    Ok(())
}

struct UnsupportedProvider;

#[async_trait]
impl LlmProvider for UnsupportedProvider {
    async fn chat(&self, _request: ChatRequest) -> anyhow::Result<ChatOutcome> {
        Ok(ChatOutcome::ServerError("unused".to_owned()))
    }

    fn model(&self) -> &str {
        "unsupported-model"
    }

    fn provider(&self) -> &'static str {
        "unsupported-test-provider"
    }
}

#[tokio::test]
async fn default_embedding_method_returns_typed_unsupported_error() -> Result<()> {
    let error = match UnsupportedProvider.embed(request(vec!["hello"])).await {
        Ok(_) => bail!("unsupported provider unexpectedly embedded input"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        EmbeddingError::Unsupported {
            provider: "unsupported-test-provider"
        }
    ));
    Ok(())
}

#[tokio::test]
async fn malformed_embedding_responses_are_rejected() -> Result<()> {
    let cases: Vec<(&str, Value, EmbeddingRequest)> = vec![
        (
            "cardinality mismatch",
            json!({
                "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                "model": "embedding-model"
            }),
            request(vec!["a", "b"]),
        ),
        (
            "duplicate index and gap",
            json!({
                "data": [
                    {"index": 0, "embedding": [1.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 4.0]}
                ],
                "model": "embedding-model"
            }),
            request(vec!["a", "b"]),
        ),
        (
            "out of range index",
            json!({
                "data": [
                    {"index": 0, "embedding": [1.0, 2.0]},
                    {"index": 2, "embedding": [3.0, 4.0]}
                ],
                "model": "embedding-model"
            }),
            request(vec!["a", "b"]),
        ),
        (
            "zero dimension",
            json!({
                "data": [{"index": 0, "embedding": []}],
                "model": "embedding-model"
            }),
            request(vec!["a"]),
        ),
        (
            "mixed dimensions",
            json!({
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 1, "embedding": [2.0, 3.0]}
                ],
                "model": "embedding-model"
            }),
            request(vec!["a", "b"]),
        ),
        (
            "requested dimension mismatch",
            json!({
                "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                "model": "embedding-model"
            }),
            request(vec!["a"]).with_dimensions(
                NonZeroU32::new(3).context("three is nonzero")?,
            ),
        ),
    ];

    for (name, body, embedding_request) in cases {
        let result = call_with_response(
            ResponseTemplate::new(200).set_body_json(body),
            embedding_request,
        )
        .await
        .with_context(|| format!("mock call for {name}"))?;
        require_invalid_response(result).with_context(|| name.to_owned())?;
    }

    let non_finite = call_with_response(
        ResponseTemplate::new(200).set_body_string(
            r#"{"data":[{"index":0,"embedding":[1e999]}],"model":"embedding-model"}"#,
        ),
        request(vec!["a"]),
    )
    .await?;
    require_invalid_response(non_finite).context("non-finite vector")?;

    let oversized_vector = vec![0.0_f32; usize::try_from(MAX_EMBEDDING_DIMENSIONS)? + 1];
    let over_limit = call_with_response(
        ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": oversized_vector}],
            "model": "embedding-model"
        })),
        request(vec!["a"]),
    )
    .await?;
    require_invalid_response(over_limit).context("over-limit vector dimension")?;
    Ok(())
}

#[tokio::test]
async fn request_and_response_bounds_are_enforced_before_unbounded_work() -> Result<()> {
    let server = MockServer::start().await;
    let provider = OpenAIProvider::with_base_url("key", "chat-model", server.uri());

    require_invalid_request(
        provider
            .embed(EmbeddingRequest::new(
                String::new(),
                vec!["hello".to_owned()],
            ))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(EmbeddingRequest::new("embedding-model", Vec::new()))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(EmbeddingRequest::new(
                "embedding-model",
                vec![String::new()],
            ))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(EmbeddingRequest::new(
                "embedding-model",
                vec![String::new(); MAX_EMBEDDING_BATCH_SIZE + 1],
            ))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(EmbeddingRequest::new(
                "embedding-model",
                vec!["x".repeat(MAX_EMBEDDING_INPUT_BYTES + 1)],
            ))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(EmbeddingRequest::new(
                "embedding-model",
                vec![
                    "x".repeat(MAX_EMBEDDING_INPUT_BYTES);
                    MAX_EMBEDDING_TOTAL_INPUT_BYTES / MAX_EMBEDDING_INPUT_BYTES + 1
                ],
            ))
            .await,
    )?;
    let oversized_dimensions =
        NonZeroU32::new(MAX_EMBEDDING_DIMENSIONS + 1).context("limit plus one is nonzero")?;
    require_invalid_request(
        provider
            .embed(request(vec!["hello"]).with_dimensions(oversized_dimensions))
            .await,
    )?;

    let received = server
        .received_requests()
        .await
        .context("mock server records requests")?;
    assert!(received.is_empty(), "invalid requests must not be dispatched");

    let oversized_length = (MAX_EMBEDDING_RESPONSE_BYTES + 1).to_string();
    let too_large = call_with_response(
        ResponseTemplate::new(200)
            .insert_header("content-length", oversized_length)
            .set_body_string("{}"),
        request(vec!["hello"]),
    )
    .await?;
    assert!(matches!(
        too_large,
        Err(EmbeddingError::ResponseTooLarge {
            limit: MAX_EMBEDDING_RESPONSE_BYTES
        })
    ));
    Ok(())
}

#[tokio::test]
async fn api_errors_are_typed_and_never_include_credentials_or_response_bodies() -> Result<()> {
    const API_KEY: &str = "sk-private-credential";
    const ECHOED_SECRET: &str = "sensitive-input-echo";
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", format!("Bearer {API_KEY}")))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_json(json!({"error": {"message": ECHOED_SECRET}})),
        )
        .mount(&server)
        .await;
    let provider = OpenAIProvider::with_base_url(API_KEY, "chat-model", server.uri());

    let error = match provider.embed(request(vec![ECHOED_SECRET])).await {
        Ok(_) => bail!("401 response unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(error, EmbeddingError::Api { status: 401 }));
    let diagnostic = format!("{error:?} {error}");
    assert!(!diagnostic.contains(API_KEY));
    assert!(!diagnostic.contains(ECHOED_SECRET));
    Ok(())
}

#[tokio::test]
async fn rate_limit_preserves_retry_after_without_exposing_the_body() -> Result<()> {
    let result = call_with_response(
        ResponseTemplate::new(429)
            .insert_header("retry-after", "7")
            .set_body_json(json!({"error": {"message": "retry later"}})),
        request(vec!["hello"]),
    )
    .await?;

    assert!(matches!(
        result,
        Err(EmbeddingError::RateLimited {
            retry_after: Some(delay)
        }) if delay == Duration::from_secs(7)
    ));
    Ok(())
}

#[cfg(feature = "gemini")]
async fn call_gemini_with_response(
    response: ResponseTemplate,
    request: EmbeddingRequest,
) -> Result<std::result::Result<EmbeddingResponse, EmbeddingError>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-004:batchEmbedContents"))
        .respond_with(response)
        .mount(&server)
        .await;
    let provider =
        GeminiProvider::new("test-key", "chat-model").with_base_url(server.uri());
    Ok(provider.embed(request).await)
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_batch_embedding_uses_current_request_shape_and_preserves_order() -> Result<()> {
    let server = MockServer::start().await;
    let dimensions = NonZeroU32::new(3).context("three is nonzero")?;
    Mock::given(method("POST"))
        .and(path("/models/gemini-embedding-001:batchEmbedContents"))
        .and(header("x-goog-api-key", "sdk-secret"))
        .and(body_json(json!({
            "requests": [
                {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": "first"}]},
                    "embedContentConfig": {"outputDimensionality": 3}
                },
                {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": "second"}]},
                    "embedContentConfig": {"outputDimensionality": 3}
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [
                {"values": [1.0, 2.0, 3.0]},
                {"values": [4.0, 5.0, 6.0]}
            ]
        })))
        .mount(&server)
        .await;
    let provider =
        GeminiProvider::new("sdk-secret", "chat-model").with_base_url(server.uri());

    let response = provider
        .embed(
            EmbeddingRequest::new(
                "models/gemini-embedding-001",
                vec!["first".to_owned(), "second".to_owned()],
            )
            .with_dimensions(dimensions),
        )
        .await
        .context("Gemini embedding request succeeds")?;

    assert_eq!(
        response.vectors,
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
    );
    Ok(())
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_rejects_unsafe_model_path_segments_before_dispatch() -> Result<()> {
    let server = MockServer::start().await;
    let provider =
        GeminiProvider::new("test-key", "chat-model").with_base_url(server.uri());
    let invalid_models = [
        "models/foo/../bar",
        "models/foo?key=value",
        "models/foo#fragment",
        "models/foo bar",
        "models/foo\nbar",
        "models/foo%2F..%2Fbar",
        "models/foo\\bar",
    ];

    for model in invalid_models {
        require_invalid_request(
            provider
                .embed(EmbeddingRequest::new(model, vec!["hello".to_owned()]))
                .await,
        )?;
    }

    let received = server
        .received_requests()
        .await
        .context("mock server records requests")?;
    assert!(received.is_empty(), "unsafe model ids must not be dispatched");
    Ok(())
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_omits_optional_dimensions_and_rejects_malformed_rows() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-004:batchEmbedContents"))
        .and(body_json(json!({
            "requests": [{
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": "hello"}]}
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [{"values": [0.25, 0.5]}]
        })))
        .mount(&server)
        .await;
    let provider =
        GeminiProvider::new("test-key", "chat-model").with_base_url(server.uri());
    let response = provider
        .embed(EmbeddingRequest::new(
            "text-embedding-004",
            vec!["hello".to_owned()],
        ))
        .await
        .context("Gemini embedding request succeeds without dimensions")?;
    assert_eq!(response.vectors, vec![vec![0.25, 0.5]]);

    let malformed = [
        json!({"embeddings": []}),
        json!({"embeddings": [{"values": [1.0]}, {"values": [2.0, 3.0]}]}),
        json!({"embeddings": [{"values": []}]}),
    ];
    for body in malformed {
        let result = call_gemini_with_response(
            ResponseTemplate::new(200).set_body_json(body),
            EmbeddingRequest::new(
                "text-embedding-004",
                vec!["first".to_owned(), "second".to_owned()],
            ),
        )
        .await?;
        require_invalid_response(result)?;
    }

    let requested_dimensions = NonZeroU32::new(3).context("three is nonzero")?;
    let wrong_dimension = call_gemini_with_response(
        ResponseTemplate::new(200)
            .set_body_json(json!({"embeddings": [{"values": [1.0, 2.0]}]})),
        EmbeddingRequest::new("text-embedding-004", vec!["hello".to_owned()])
            .with_dimensions(requested_dimensions),
    )
    .await?;
    require_invalid_response(wrong_dimension)?;

    let non_finite = call_gemini_with_response(
        ResponseTemplate::new(200)
            .set_body_string(r#"{"embeddings":[{"values":[1e999]}]}"#),
        EmbeddingRequest::new("text-embedding-004", vec!["hello".to_owned()]),
    )
    .await?;
    require_invalid_response(non_finite)
}