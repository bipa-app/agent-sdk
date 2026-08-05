#![cfg(feature = "openai")]

use std::num::NonZeroU32;
use std::time::Duration;

#[cfg(feature = "gemini")]
use agent_sdk_providers::GeminiProvider;
use agent_sdk_providers::{
    ChatOutcome, ChatRequest, EmbeddingError, EmbeddingRequest, EmbeddingResponse, LlmProvider,
    MAX_EMBEDDING_BATCH_SIZE, MAX_EMBEDDING_DIMENSIONS, MAX_EMBEDDING_INPUT_BYTES,
    MAX_EMBEDDING_MODEL_BYTES, MAX_EMBEDDING_RESPONSE_BYTES, MAX_EMBEDDING_TOTAL_INPUT_BYTES,
    OpenAIProvider, validate_embedding_request, validate_embedding_response,
};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request(inputs: Vec<&str>) -> EmbeddingRequest {
    EmbeddingRequest::new(
        "text-embedding-3-large",
        inputs.into_iter().map(str::to_owned).collect(),
    )
}

#[test]
fn reusable_embedding_validators_are_exported_for_custom_providers() -> Result<()> {
    let dimensions = NonZeroU32::new(2).context("two is nonzero")?;
    let request = request(vec!["first", "second"]).with_dimensions(dimensions);
    validate_embedding_request(&request).context("request is valid")?;

    let response = EmbeddingResponse {
        vectors: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    };
    validate_embedding_response(&request, &response).context("response is valid")
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
    Ok(provider.embed(&request).await)
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
async fn openai_embedding_request_reuses_base_url_auth_headers_and_restores_input_order()
-> Result<()> {
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
        .embed(&request(vec!["first", "second"]).with_dimensions(dimensions))
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
        .embed(&EmbeddingRequest::new(
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

    fn model(&self) -> &'static str {
        "unsupported-model"
    }

    fn provider(&self) -> &'static str {
        "unsupported-test-provider"
    }
}

#[tokio::test]
async fn default_embedding_method_returns_typed_unsupported_error() -> Result<()> {
    let Err(error) = UnsupportedProvider.embed(&request(vec!["hello"])).await else {
        bail!("unsupported provider unexpectedly embedded input")
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
            request(vec!["a"]).with_dimensions(NonZeroU32::new(3).context("three is nonzero")?),
        ),
        (
            "empty response model",
            json!({
                "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                "model": " "
            }),
            request(vec!["a"]),
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

    Ok(())
}

#[tokio::test]
async fn over_dimension_and_row_limits_reject_compact_json_during_parsing() -> Result<()> {
    let vector_values = format!(
        "{}0",
        "0,".repeat(usize::try_from(MAX_EMBEDDING_DIMENSIONS)?)
    );
    let over_dimension = call_with_response(
        ResponseTemplate::new(200).set_body_string(format!(
            r#"{{"data":[{{"index":0,"embedding":[{vector_values}]}}],"model":"embedding-model"}}"#
        )),
        request(vec!["a"]),
    )
    .await?;
    require_invalid_response(over_dimension).context("over-limit vector dimension")?;

    let row = r#"{"index":0,"embedding":[0.0]}"#;
    let rows = format!("{row},").repeat(MAX_EMBEDDING_BATCH_SIZE) + row;
    let over_rows = call_with_response(
        ResponseTemplate::new(200)
            .set_body_string(format!(r#"{{"data":[{rows}],"model":"embedding-model"}}"#)),
        EmbeddingRequest::new(
            "embedding-model",
            vec!["a".to_owned(); MAX_EMBEDDING_BATCH_SIZE],
        ),
    )
    .await?;
    require_invalid_response(over_rows).context("over-limit embedding row count")
}

#[tokio::test]
async fn request_bounds_are_rejected_without_dispatch() -> Result<()> {
    let server = MockServer::start().await;
    let provider = OpenAIProvider::with_base_url("key", "chat-model", server.uri());

    require_invalid_request(
        provider
            .embed(&EmbeddingRequest::new(
                String::new(),
                vec!["hello".to_owned()],
            ))
            .await,
    )?;
    let overlong_model = "é".repeat(MAX_EMBEDDING_MODEL_BYTES / "é".len() + 1);
    let overlong_model_result = provider
        .embed(&EmbeddingRequest::new(
            overlong_model,
            vec!["hello".to_owned()],
        ))
        .await;
    match &overlong_model_result {
        Err(EmbeddingError::InvalidRequest { message }) => assert!(
            message.contains(&MAX_EMBEDDING_MODEL_BYTES.to_string()),
            "model-bound diagnostic must include the public byte limit"
        ),
        Ok(_) => bail!("overlong embedding model unexpectedly succeeded"),
        Err(error) => bail!("expected invalid-request error, got {error:?}"),
    }
    require_invalid_request(overlong_model_result)?;
    require_invalid_request(
        provider
            .embed(&EmbeddingRequest::new("embedding-model", Vec::new()))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(&EmbeddingRequest::new(
                "embedding-model",
                vec![String::new()],
            ))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(&EmbeddingRequest::new(
                "embedding-model",
                vec![String::new(); MAX_EMBEDDING_BATCH_SIZE + 1],
            ))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(&EmbeddingRequest::new(
                "embedding-model",
                vec!["x".repeat(MAX_EMBEDDING_INPUT_BYTES + 1)],
            ))
            .await,
    )?;
    require_invalid_request(
        provider
            .embed(&EmbeddingRequest::new(
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
            .embed(&request(vec!["hello"]).with_dimensions(oversized_dimensions))
            .await,
    )?;

    let received = server
        .received_requests()
        .await
        .context("mock server records requests")?;
    assert!(
        received.is_empty(),
        "invalid requests must not be dispatched"
    );
    Ok(())
}

#[tokio::test]
async fn oversized_content_length_is_rejected_before_body_read() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let declared_length = MAX_EMBEDDING_RESPONSE_BYTES + 1;
    let server_task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await?;
        let mut request_byte = [0_u8; 1];
        socket.read_exact(&mut request_byte).await?;
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {declared_length}\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            )
            .await?;
        Ok::<(), std::io::Error>(())
    });
    let provider = OpenAIProvider::with_base_url("key", "chat-model", format!("http://{address}"));

    let result = provider.embed(&request(vec!["hello"])).await;

    server_task.await.context("raw HTTP server task joins")??;
    assert!(matches!(
        result,
        Err(EmbeddingError::ResponseTooLarge {
            limit: MAX_EMBEDDING_RESPONSE_BYTES
        })
    ));
    Ok(())
}

async fn call_with_stalled_error_body(
    status: &str,
    headers: &str,
) -> Result<std::result::Result<EmbeddingResponse, EmbeddingError>> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let declared_length = MAX_EMBEDDING_RESPONSE_BYTES + 1;
    let response_headers =
        format!("HTTP/1.1 {status}\r\nContent-Length: {declared_length}\r\n{headers}\r\n");
    let server_task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await?;
        let mut request_byte = [0_u8; 1];
        socket.read_exact(&mut request_byte).await?;
        socket.write_all(response_headers.as_bytes()).await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
        Ok::<(), std::io::Error>(())
    });
    let provider = OpenAIProvider::with_base_url("key", "chat-model", format!("http://{address}"));

    let timed = tokio::time::timeout(
        Duration::from_millis(200),
        provider.embed(&request(vec!["hello"])),
    )
    .await;
    server_task.abort();
    timed.context("status classification must not wait for the error body")
}

#[tokio::test]
async fn non_rate_limit_api_status_does_not_consume_the_body() -> Result<()> {
    let result = call_with_stalled_error_body("401 Unauthorized", "").await?;
    assert!(matches!(result, Err(EmbeddingError::Api { status: 401 })));
    Ok(())
}

#[tokio::test]
async fn retry_after_rate_limit_does_not_consume_the_body() -> Result<()> {
    let result =
        call_with_stalled_error_body("429 Too Many Requests", "Retry-After: 7\r\n").await?;
    assert!(matches!(
        result,
        Err(EmbeddingError::RateLimited {
            retry_after: Some(delay)
        }) if delay == Duration::from_secs(7)
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
            ResponseTemplate::new(401).set_body_json(json!({"error": {"message": ECHOED_SECRET}})),
        )
        .mount(&server)
        .await;
    let provider = OpenAIProvider::with_base_url(API_KEY, "chat-model", server.uri());

    let Err(error) = provider.embed(&request(vec![ECHOED_SECRET])).await else {
        bail!("401 response unexpectedly succeeded")
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

#[tokio::test]
async fn openai_headerless_rate_limit_uses_retry_hint_from_bounded_prefix() -> Result<()> {
    let result = call_with_response(
        ResponseTemplate::new(429).set_body_string("Try again in 7s."),
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

#[tokio::test]
async fn openai_malformed_retry_after_falls_back_to_bounded_body_hint() -> Result<()> {
    let result = call_with_response(
        ResponseTemplate::new(429)
            .insert_header("retry-after", "not-a-delay")
            .set_body_string("Try again in 7s."),
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

#[tokio::test]
async fn openai_embedding_retry_hint_beyond_scan_prefix_is_ignored() -> Result<()> {
    let body = format!("{}Try again in 7s.", "x".repeat(64 * 1024));
    let result = call_with_response(
        ResponseTemplate::new(429).set_body_string(body),
        request(vec!["hello"]),
    )
    .await?;

    assert!(matches!(
        result,
        Err(EmbeddingError::RateLimited { retry_after: None })
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
    let provider = GeminiProvider::new("test-key", "chat-model").with_base_url(server.uri());
    Ok(provider.embed(&request).await)
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_embedding_retry_hint_beyond_scan_prefix_is_ignored() -> Result<()> {
    let body = json!({
        "error": {
            "message": "x".repeat(64 * 1024),
            "details": [{
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "7s"
            }]
        }
    });
    let result = call_gemini_with_response(
        ResponseTemplate::new(429).set_body_json(body),
        EmbeddingRequest::new("text-embedding-004", vec!["hello".to_owned()]),
    )
    .await?;

    assert!(matches!(
        result,
        Err(EmbeddingError::RateLimited { retry_after: None })
    ));
    Ok(())
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_headerless_rate_limit_uses_retry_hint_from_bounded_prefix() -> Result<()> {
    let result = call_gemini_with_response(
        ResponseTemplate::new(429).set_body_json(json!({
            "error": {
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "7s"
                }]
            }
        })),
        EmbeddingRequest::new("text-embedding-004", vec!["hello".to_owned()]),
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
    let provider = GeminiProvider::new("sdk-secret", "chat-model").with_base_url(server.uri());

    let response = provider
        .embed(
            &EmbeddingRequest::new(
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
    let provider = GeminiProvider::new("test-key", "chat-model").with_base_url(server.uri());
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
                .embed(&EmbeddingRequest::new(model, vec!["hello".to_owned()]))
                .await,
        )?;
    }

    let received = server
        .received_requests()
        .await
        .context("mock server records requests")?;
    assert!(
        received.is_empty(),
        "unsafe model ids must not be dispatched"
    );
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
    let provider = GeminiProvider::new("test-key", "chat-model").with_base_url(server.uri());
    let response = provider
        .embed(&EmbeddingRequest::new(
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
        ResponseTemplate::new(200).set_body_json(json!({"embeddings": [{"values": [1.0, 2.0]}]})),
        EmbeddingRequest::new("text-embedding-004", vec!["hello".to_owned()])
            .with_dimensions(requested_dimensions),
    )
    .await?;
    require_invalid_response(wrong_dimension)?;

    let non_finite = call_gemini_with_response(
        ResponseTemplate::new(200).set_body_string(r#"{"embeddings":[{"values":[1e999]}]}"#),
        EmbeddingRequest::new("text-embedding-004", vec!["hello".to_owned()]),
    )
    .await?;
    require_invalid_response(non_finite)
}
