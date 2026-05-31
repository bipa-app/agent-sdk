//! Streamable-HTTP (and SSE) MCP transport.
//!
//! Implements the MCP "Streamable HTTP" transport introduced in revision
//! `2025-03-26` and carried forward in later revisions. A single HTTP endpoint
//! serves every JSON-RPC message:
//!
//! * The client `POST`s a JSON-RPC request (or notification) to the endpoint.
//! * The server replies with either a single `application/json` body (one
//!   JSON-RPC message) or a `text/event-stream` body (Server-Sent Events, each
//!   `data:` line carrying one JSON-RPC message). Either way, this transport
//!   resolves the [`JsonRpcResponse`] whose `id` matches the request it sent.
//! * The server may issue a `Mcp-Session-Id` header on the `initialize`
//!   response; the client echoes it on all subsequent requests.
//! * After initialization the client sends the negotiated `MCP-Protocol-Version`
//!   header on every request, as the spec mandates.
//!
//! Authentication is supplied as a bearer token / OAuth access token (sent as an
//! `Authorization: Bearer …` header) or arbitrary custom headers.
//!
//! The transport is generic over the HTTP layer via [`HttpPoster`]: production
//! code uses [`ReqwestPoster`] (the default), while tests inject a scripted
//! poster to exercise the JSON and SSE response paths deterministically with no
//! live network.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

use super::protocol::{JsonRpcRequest, JsonRpcResponse, RequestId};
use super::transport::McpTransport;

/// Header carrying the MCP session id assigned by the server.
const SESSION_ID_HEADER: &str = "Mcp-Session-Id";
/// Header carrying the negotiated MCP protocol revision.
const PROTOCOL_VERSION_HEADER: &str = "MCP-Protocol-Version";

/// A single HTTP response from an MCP endpoint, normalised across the two
/// streamable-HTTP body shapes.
#[derive(Clone, Debug)]
pub struct HttpReply {
    /// `Content-Type` of the response body (lower-cased, no parameters).
    pub content_type: String,
    /// Raw response body bytes.
    pub body: String,
    /// Value of the `Mcp-Session-Id` response header, if present.
    pub session_id: Option<String>,
}

impl HttpReply {
    /// Construct a JSON-body reply (`application/json`).
    #[must_use]
    pub fn json(body: impl Into<String>) -> Self {
        Self {
            content_type: "application/json".to_string(),
            body: body.into(),
            session_id: None,
        }
    }

    /// Construct an SSE-body reply (`text/event-stream`).
    #[must_use]
    pub fn event_stream(body: impl Into<String>) -> Self {
        Self {
            content_type: "text/event-stream".to_string(),
            body: body.into(),
            session_id: None,
        }
    }

    /// Attach a session id to this reply (as if returned in the header).
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

/// The HTTP request a transport wants to make, in transport-neutral form.
#[derive(Clone, Debug)]
pub struct HttpRequest {
    /// Serialized JSON-RPC body to POST.
    pub body: String,
    /// `Authorization` header value, if a token is configured.
    pub authorization: Option<String>,
    /// `Mcp-Session-Id` to echo, once one has been assigned.
    pub session_id: Option<String>,
    /// Negotiated `MCP-Protocol-Version`, once initialization has completed.
    pub protocol_version: Option<String>,
    /// Extra static headers configured on the transport.
    pub extra_headers: Vec<(String, String)>,
}

/// Abstraction over the act of `POST`ing one JSON-RPC message to the MCP endpoint.
///
/// Production uses [`ReqwestPoster`]; tests inject a scripted poster so the
/// JSON / SSE decode paths run with zero live network.
#[async_trait]
pub trait HttpPoster: Send + Sync {
    /// POST `request` to the MCP endpoint and return the normalised reply.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the server returns a
    /// non-success status.
    async fn post(&self, request: HttpRequest) -> Result<HttpReply>;
}

/// Authentication strategy for an HTTP MCP connection.
#[derive(Clone, Debug, Default)]
pub enum McpAuth {
    /// No authentication.
    #[default]
    None,
    /// Static bearer token / OAuth access token sent as `Authorization: Bearer`.
    Bearer(String),
}

impl McpAuth {
    /// Render the `Authorization` header value, if any.
    #[must_use]
    fn header_value(&self) -> Option<String> {
        match self {
            Self::None => None,
            Self::Bearer(token) => Some(format!("Bearer {token}")),
        }
    }
}

/// Streamable-HTTP MCP transport.
///
/// Construct with [`StreamableHttpTransport::new`] for a live connection, or
/// [`StreamableHttpTransport::with_poster`] to inject a custom [`HttpPoster`]
/// (used by tests).
pub struct StreamableHttpTransport {
    poster: Arc<dyn HttpPoster>,
    auth: McpAuth,
    extra_headers: Vec<(String, String)>,
    next_id: AtomicU64,
    /// Session id assigned by the server on `initialize`, echoed thereafter.
    session_id: RwLock<Option<String>>,
    /// Protocol revision negotiated during `initialize`.
    protocol_version: RwLock<Option<String>>,
}

impl StreamableHttpTransport {
    /// Create a transport that talks to `endpoint` over real HTTP.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying HTTP client cannot be built.
    pub fn new(endpoint: impl Into<String>, auth: McpAuth) -> Result<Arc<Self>> {
        let poster = ReqwestPoster::new(endpoint)?;
        Ok(Self::with_poster(Arc::new(poster), auth))
    }

    /// Create a transport backed by a custom [`HttpPoster`].
    ///
    /// This is the seam tests use to script JSON / SSE responses without a
    /// network.
    #[must_use]
    pub fn with_poster(poster: Arc<dyn HttpPoster>, auth: McpAuth) -> Arc<Self> {
        Arc::new(Self {
            poster,
            auth,
            extra_headers: Vec::new(),
            next_id: AtomicU64::new(1),
            session_id: RwLock::new(None),
            protocol_version: RwLock::new(None),
        })
    }

    /// Add a static custom header sent on every request (e.g. a tenant id).
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    fn next_request_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    async fn build_http_request(&self, body: String) -> HttpRequest {
        HttpRequest {
            body,
            authorization: self.auth.header_value(),
            session_id: self.session_id.read().await.clone(),
            protocol_version: self.protocol_version.read().await.clone(),
            extra_headers: self.extra_headers.clone(),
        }
    }

    /// Capture the session id from a reply if the server assigned one.
    async fn capture_session_id(&self, reply: &HttpReply) {
        if let Some(ref sid) = reply.session_id {
            let mut guard = self.session_id.write().await;
            if guard.as_deref() != Some(sid.as_str()) {
                *guard = Some(sid.clone());
            }
        }
    }
}

/// Parse a normalised [`HttpReply`] into the JSON-RPC response matching `id`.
///
/// Handles both the single-JSON body and the SSE multi-event body. For SSE, the
/// first `data:` payload that parses as a [`JsonRpcResponse`] whose `id` matches
/// the request is returned; intervening server-initiated notifications/requests
/// (which carry no matching `id`) are skipped.
fn parse_reply(reply: &HttpReply, id: &RequestId) -> Result<JsonRpcResponse> {
    if reply.content_type.contains("text/event-stream") {
        parse_sse_response(&reply.body, id)
    } else {
        serde_json::from_str::<JsonRpcResponse>(reply.body.trim())
            .context("failed to parse JSON MCP response body")
    }
}

/// Extract the matching JSON-RPC response from an SSE body.
fn parse_sse_response(body: &str, id: &RequestId) -> Result<JsonRpcResponse> {
    let mut data_buf = String::new();
    let mut last_parsed: Option<JsonRpcResponse> = None;

    let flush =
        |data: &mut String, last: &mut Option<JsonRpcResponse>| -> Option<JsonRpcResponse> {
            if data.is_empty() {
                return None;
            }
            let raw = std::mem::take(data);
            if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(raw.trim()) {
                if &resp.id == id {
                    return Some(resp);
                }
                *last = Some(resp);
            }
            None
        };

    for line in body.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            // Event boundary: attempt to resolve the accumulated data block.
            if let Some(resp) = flush(&mut data_buf, &mut last_parsed) {
                return Ok(resp);
            }
            continue;
        }
        // SSE `data:` lines (optionally with a leading space) carry the payload.
        if let Some(rest) = line.strip_prefix("data:") {
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            if !data_buf.is_empty() {
                data_buf.push('\n');
            }
            data_buf.push_str(rest);
        }
        // Other SSE fields (`event:`, `id:`, comments) are ignored.
    }
    // Flush any trailing event with no terminating blank line.
    if let Some(resp) = flush(&mut data_buf, &mut last_parsed) {
        return Ok(resp);
    }

    last_parsed.context("SSE stream contained no JSON-RPC response matching the request id")
}

#[async_trait]
impl McpTransport for StreamableHttpTransport {
    async fn send(&self, mut request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let id = self.next_request_id();
        request.id = RequestId::Number(id);
        let request_id = request.id.clone();

        let body = serde_json::to_string(&request).context("failed to serialize MCP request")?;
        let http_request = self.build_http_request(body).await;
        let reply = self.poster.post(http_request).await?;
        self.capture_session_id(&reply).await;

        let response = parse_reply(&reply, &request_id)?;

        if let Some(ref error) = response.error {
            bail!("JSON-RPC error {}: {}", error.code, error.message);
        }
        Ok(response)
    }

    async fn send_notification(&self, mut request: JsonRpcRequest) -> Result<()> {
        // Notifications carry no id per JSON-RPC, but our request type requires
        // one for serialization; the server ignores it for notification methods.
        let id = self.next_request_id();
        request.id = RequestId::Number(id);
        let body = serde_json::to_string(&request).context("failed to serialize MCP request")?;
        let http_request = self.build_http_request(body).await;
        let reply = self.poster.post(http_request).await?;
        self.capture_session_id(&reply).await;
        Ok(())
    }

    async fn set_protocol_version(&self, version: &str) {
        let mut guard = self.protocol_version.write().await;
        *guard = Some(version.to_string());
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// Default [`HttpPoster`] backed by `reqwest`.
pub struct ReqwestPoster {
    client: reqwest::Client,
    endpoint: String,
}

impl ReqwestPoster {
    /// Build a reqwest-backed poster for `endpoint`.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be constructed.
    pub fn new(endpoint: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .build()
            .context("failed to build MCP HTTP client")?;
        Ok(Self {
            client,
            endpoint: endpoint.into(),
        })
    }

    /// Build a poster from a caller-supplied `reqwest::Client`.
    #[must_use]
    pub fn with_client(client: reqwest::Client, endpoint: impl Into<String>) -> Self {
        Self {
            client,
            endpoint: endpoint.into(),
        }
    }
}

#[async_trait]
impl HttpPoster for ReqwestPoster {
    async fn post(&self, request: HttpRequest) -> Result<HttpReply> {
        let mut builder = self
            .client
            .post(&self.endpoint)
            // The streamable-HTTP spec requires the client to accept both shapes.
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(request.body);

        if let Some(auth) = request.authorization {
            builder = builder.header(reqwest::header::AUTHORIZATION, auth);
        }
        if let Some(sid) = request.session_id {
            builder = builder.header(SESSION_ID_HEADER, sid);
        }
        if let Some(version) = request.protocol_version {
            builder = builder.header(PROTOCOL_VERSION_HEADER, version);
        }
        for (name, value) in request.extra_headers {
            builder = builder.header(name, value);
        }

        let response = builder
            .send()
            .await
            .context("MCP HTTP request failed to send")?;

        let status = response.status();
        let session_id = response
            .headers()
            .get(SESSION_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map_or_else(
                || "application/json".to_string(),
                |s| s.split(';').next().unwrap_or(s).trim().to_lowercase(),
            );

        let body = response
            .text()
            .await
            .context("failed to read MCP HTTP response body")?;

        if !status.is_success() {
            bail!("MCP HTTP request returned status {status}: {body}");
        }

        Ok(HttpReply {
            content_type,
            body,
            session_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_response(id: u64, result: &serde_json::Value) -> String {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        })
        .to_string()
    }

    #[test]
    fn parse_json_body() {
        let reply = HttpReply::json(ok_response(1, &serde_json::json!({"ok": true})));
        let resp = parse_reply(&reply, &RequestId::Number(1)).expect("parse");
        assert!(!resp.is_error());
        assert!(resp.result().is_some());
    }

    #[test]
    fn parse_sse_single_event() {
        let body = format!(
            "event: message\ndata: {}\n\n",
            ok_response(2, &serde_json::json!({}))
        );
        let reply = HttpReply::event_stream(body);
        let resp = parse_reply(&reply, &RequestId::Number(2)).expect("parse");
        assert_eq!(resp.id, RequestId::Number(2));
    }

    #[test]
    fn parse_sse_skips_non_matching_then_matches() {
        // A server-initiated notification-shaped message (id 99) precedes the
        // real response (id 3); the parser must skip ahead to the match.
        let body = format!(
            "data: {}\n\ndata: {}\n\n",
            ok_response(99, &serde_json::json!({"unrelated": true})),
            ok_response(3, &serde_json::json!({"answer": 42})),
        );
        let reply = HttpReply::event_stream(body);
        let resp = parse_reply(&reply, &RequestId::Number(3)).expect("parse");
        assert_eq!(resp.id, RequestId::Number(3));
    }

    #[test]
    fn parse_sse_multiline_data() {
        // SSE allows a payload to be split across consecutive `data:` lines,
        // re-joined with newlines.
        let body = "data: {\"jsonrpc\":\"2.0\",\ndata: \"id\":4,\ndata: \"result\":{}}\n\n";
        let reply = HttpReply::event_stream(body.to_string());
        let resp = parse_reply(&reply, &RequestId::Number(4)).expect("parse");
        assert_eq!(resp.id, RequestId::Number(4));
    }

    #[test]
    fn bearer_auth_header_value() {
        assert_eq!(McpAuth::None.header_value(), None);
        assert_eq!(
            McpAuth::Bearer("tok".to_string()).header_value().as_deref(),
            Some("Bearer tok"),
        );
    }
}
