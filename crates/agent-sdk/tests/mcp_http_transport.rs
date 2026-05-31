//! Deterministic integration tests for the streamable-HTTP MCP transport.
//!
//! These mirror the existing `ScriptedMcpTransport` test double, but drive the
//! real [`StreamableHttpTransport`] through a scripted [`HttpPoster`] so the
//! end-to-end client flow (initialize → tools/list → tools/call → resources →
//! prompts) is exercised over the HTTP/SSE decode paths with zero live network.

#![cfg(feature = "mcp")]

use std::sync::Arc;
use std::sync::Mutex;

use agent_sdk::mcp::{
    HttpPoster, HttpReply, HttpRequest, McpAuth, McpClient, StreamableHttpTransport,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};

/// A scripted HTTP MCP server.
///
/// Each `post` returns the next canned reply (FIFO) and records the request it
/// received so tests can assert on the headers/body the transport sent.
struct ScriptedHttpServer {
    replies: Mutex<std::collections::VecDeque<HttpReply>>,
    seen: Mutex<Vec<HttpRequest>>,
}

impl ScriptedHttpServer {
    fn new(replies: Vec<HttpReply>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into_iter().collect()),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.seen.lock().expect("seen lock").clone()
    }
}

#[async_trait]
impl HttpPoster for ScriptedHttpServer {
    async fn post(&self, request: HttpRequest) -> Result<HttpReply> {
        self.seen.lock().expect("seen lock").push(request);
        self.replies
            .lock()
            .expect("replies lock")
            .pop_front()
            .ok_or_else(|| anyhow!("ScriptedHttpServer ran out of replies"))
    }
}

/// JSON-RPC success envelope as an `application/json` reply. The transport
/// rewrites the request id, so the id we put here is overwritten on `send`;
/// the SSE path however requires a matching id, so we set it to the id the
/// transport will assign (the request counter starts at 1).
fn json_result(id: u64, result: &Value) -> HttpReply {
    HttpReply::json(json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string())
}

fn sse_result(id: u64, result: &Value) -> HttpReply {
    let payload = json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string();
    HttpReply::event_stream(format!("event: message\ndata: {payload}\n\n"))
}

fn initialize_result(version: &str) -> Value {
    json!({
        "protocolVersion": version,
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": {},
            "prompts": {}
        },
        "serverInfo": { "name": "scripted-http", "version": "1.0.0" }
    })
}

#[tokio::test]
async fn connects_over_http_and_calls_tools_with_bearer_auth() -> Result<()> {
    // initialize → tools/list → tools/call, all over JSON bodies.
    // The transport assigns ids 1, 2, 3 to the three `send()` calls (the
    // `notifications/initialized` notification consumes id between them but
    // produces no reply we script here — wait: notification IS a POST, so it
    // consumes a reply). Order of POSTs: initialize(1), initialized-notif(2),
    // tools/list(3), tools/call(4).
    let server = ScriptedHttpServer::new(vec![
        json_result(1, &initialize_result("2025-06-18")).with_session_id("sess-123"),
        // notifications/initialized — server ack (empty 200 body is fine).
        HttpReply::json("{}"),
        json_result(
            3,
            &json!({ "tools": [{
                "name": "echo",
                "description": "Echoes input",
                "inputSchema": { "type": "object" }
            }] }),
        ),
        json_result(
            4,
            &json!({ "content": [{ "type": "text", "text": "pong" }], "isError": false }),
        ),
    ]);

    let transport = StreamableHttpTransport::with_poster(
        Arc::clone(&server) as Arc<dyn HttpPoster>,
        McpAuth::Bearer("secret-token".to_string()),
    );

    let client = McpClient::new(transport, "remote".to_string()).await?;

    // Acceptance: protocol negotiation uses a current revision, not pinned.
    assert_eq!(client.protocol_version(), Some("2025-06-18"));

    let tools = client.list_tools().await?;
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");

    let result = client.call_tool("echo", json!({ "msg": "ping" })).await?;
    assert!(!result.is_error);

    // The initialize POST must carry the bearer token.
    let requests = server.requests();
    assert!(!requests.is_empty());
    assert_eq!(
        requests[0].authorization.as_deref(),
        Some("Bearer secret-token"),
        "initialize must send the OAuth/bearer Authorization header",
    );

    // Once the session id was assigned on initialize, later POSTs must echo it.
    let tools_call = requests
        .last()
        .ok_or_else(|| anyhow!("no requests recorded"))?;
    assert_eq!(tools_call.session_id.as_deref(), Some("sess-123"));
    // And the negotiated protocol version must be carried out-of-band.
    assert_eq!(tools_call.protocol_version.as_deref(), Some("2025-06-18"));

    Ok(())
}

#[tokio::test]
async fn honours_server_selected_legacy_protocol_revision() -> Result<()> {
    // A legacy server downgrades to 2024-11-05; the client must adapt.
    let server = ScriptedHttpServer::new(vec![
        json_result(1, &initialize_result("2024-11-05")),
        HttpReply::json("{}"),
    ]);
    let transport = StreamableHttpTransport::with_poster(
        Arc::clone(&server) as Arc<dyn HttpPoster>,
        McpAuth::None,
    );
    let client = McpClient::new(transport, "legacy".to_string()).await?;
    assert_eq!(client.protocol_version(), Some("2024-11-05"));
    Ok(())
}

#[tokio::test]
async fn decodes_tool_call_over_sse() -> Result<()> {
    // The tools/call response arrives as a text/event-stream body.
    let server = ScriptedHttpServer::new(vec![
        json_result(1, &initialize_result("2025-06-18")),
        HttpReply::json("{}"),
        sse_result(
            3,
            &json!({ "content": [{ "type": "text", "text": "streamed" }], "isError": false }),
        ),
    ]);
    let transport = StreamableHttpTransport::with_poster(
        Arc::clone(&server) as Arc<dyn HttpPoster>,
        McpAuth::None,
    );
    let client = McpClient::new(transport, "sse".to_string()).await?;
    let result = client.call_tool("noop", json!({})).await?;
    assert!(!result.is_error);
    assert_eq!(
        result.content.len(),
        1,
        "SSE-decoded tool result should carry one content block",
    );
    Ok(())
}

#[tokio::test]
async fn lists_resources_and_reads_one() -> Result<()> {
    let server = ScriptedHttpServer::new(vec![
        json_result(1, &initialize_result("2025-06-18")),
        HttpReply::json("{}"),
        json_result(
            3,
            &json!({ "resources": [{
                "uri": "file:///readme.md",
                "name": "Readme",
                "mimeType": "text/markdown"
            }] }),
        ),
        json_result(
            4,
            &json!({ "contents": [{
                "uri": "file:///readme.md",
                "mimeType": "text/markdown",
                "text": "# Hello"
            }] }),
        ),
    ]);
    let transport = StreamableHttpTransport::with_poster(
        Arc::clone(&server) as Arc<dyn HttpPoster>,
        McpAuth::None,
    );
    let client = McpClient::new(transport, "res".to_string()).await?;
    assert!(client.supports_resources());

    let resources = client.list_resources().await?;
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0].uri, "file:///readme.md");

    let read = client.read_resource("file:///readme.md").await?;
    assert_eq!(read.contents.len(), 1);
    assert_eq!(read.contents[0].text.as_deref(), Some("# Hello"));
    Ok(())
}

#[tokio::test]
async fn lists_prompts_and_renders_one() -> Result<()> {
    let server = ScriptedHttpServer::new(vec![
        json_result(1, &initialize_result("2025-06-18")),
        HttpReply::json("{}"),
        json_result(
            3,
            &json!({ "prompts": [{
                "name": "greet",
                "description": "A greeting",
                "arguments": [{ "name": "name", "required": true }]
            }] }),
        ),
        json_result(
            4,
            &json!({
                "description": "rendered greeting",
                "messages": [{
                    "role": "user",
                    "content": { "type": "text", "text": "Hello, Ada" }
                }]
            }),
        ),
    ]);
    let transport = StreamableHttpTransport::with_poster(
        Arc::clone(&server) as Arc<dyn HttpPoster>,
        McpAuth::None,
    );
    let client = McpClient::new(transport, "prompts".to_string()).await?;
    assert!(client.supports_prompts());

    let prompts = client.list_prompts().await?;
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0].name, "greet");
    assert!(prompts[0].arguments[0].required);

    let rendered = client
        .get_prompt("greet", Some(json!({ "name": "Ada" })))
        .await?;
    assert_eq!(rendered.messages.len(), 1);
    Ok(())
}

#[tokio::test]
async fn resources_and_prompts_skip_when_uncapable() -> Result<()> {
    // Server advertises neither resources nor prompts: list calls short-circuit
    // to empty without a network round-trip (so we script no extra replies).
    let server = ScriptedHttpServer::new(vec![
        HttpReply::json(
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "minimal", "version": "1.0.0" }
                }
            })
            .to_string(),
        ),
        HttpReply::json("{}"),
    ]);
    let transport = StreamableHttpTransport::with_poster(
        Arc::clone(&server) as Arc<dyn HttpPoster>,
        McpAuth::None,
    );
    let client = McpClient::new(transport, "minimal".to_string()).await?;
    assert!(!client.supports_resources());
    assert!(!client.supports_prompts());
    assert!(client.list_resources().await?.is_empty());
    assert!(client.list_prompts().await?.is_empty());
    Ok(())
}
