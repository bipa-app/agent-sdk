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
use std::time::Duration;
use tokio::sync::RwLock;

use super::protocol::{JsonRpcRequest, JsonRpcResponse, RequestId};
use super::transport::{McpTransport, notification_body};

/// Header carrying the MCP session id assigned by the server.
const SESSION_ID_HEADER: &str = "Mcp-Session-Id";
/// Header carrying the negotiated MCP protocol revision.
const PROTOCOL_VERSION_HEADER: &str = "MCP-Protocol-Version";

/// Default request timeout for the reqwest client backing [`ReqwestPoster`].
///
/// Without this, a streamable-HTTP server that holds an SSE stream open (with
/// keep-alive comments) would block a request forever. Matches the stdio
/// transport's default response timeout. Override per poster with
/// [`ReqwestPoster::with_timeout`].
pub const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_mins(1);

/// Default per-request send deadline for [`StreamableHttpTransport`] (60s).
///
/// Applied around each [`HttpPoster::post`] call, independent of the underlying
/// client's own timeout, so `send`/`send_notification` always have a
/// cancellation path. Override per transport with
/// [`StreamableHttpTransport::with_request_timeout`] or
/// [`StreamableHttpTransport::with_timeout`].
pub const DEFAULT_SEND_DEADLINE: Duration = Duration::from_mins(1);

/// Maximum response body the [`ReqwestPoster`] will buffer. An endless SSE
/// stream would otherwise grow memory without bound.
const MAX_RESPONSE_BODY_BYTES: usize = 16 * 1024 * 1024;

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
    /// Overall deadline applied around each [`HttpPoster::post`] call so a
    /// slow, hung, or keep-alive SSE server can never wedge a turn. Defaults to
    /// [`DEFAULT_SEND_DEADLINE`]; set via [`StreamableHttpTransport::with_request_timeout`].
    send_deadline: Duration,
}

impl StreamableHttpTransport {
    /// Create a transport that talks to `endpoint` over real HTTP.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying HTTP client cannot be built.
    pub fn new(endpoint: impl Into<String>, auth: McpAuth) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::builder(endpoint, auth)?))
    }

    /// Create a transport over real HTTP with a custom per-request timeout.
    ///
    /// Sets *both* the underlying reqwest client's request timeout and the
    /// transport-level send deadline to `request_timeout`, so a slow or hung
    /// streamable-HTTP server trips this deadline instead of the
    /// [`DEFAULT_SEND_DEADLINE`] / [`DEFAULT_HTTP_TIMEOUT`] defaults. MCP tool
    /// calls routinely exceed 60s (builds, codegen); raise the timeout for
    /// those servers, or lower it for latency-sensitive ones.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying HTTP client cannot be built.
    pub fn with_timeout(
        endpoint: impl Into<String>,
        auth: McpAuth,
        request_timeout: Duration,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::builder_with_timeout(
            endpoint,
            auth,
            request_timeout,
        )?))
    }

    /// Create a transport backed by a custom [`HttpPoster`].
    ///
    /// This is the seam tests use to script JSON / SSE responses without a
    /// network.
    #[must_use]
    pub fn with_poster(poster: Arc<dyn HttpPoster>, auth: McpAuth) -> Arc<Self> {
        Arc::new(Self::with_poster_owned(poster, auth))
    }

    /// Create an un-wrapped transport over real HTTP for further builder-style
    /// configuration (e.g. [`StreamableHttpTransport::with_header`]).
    ///
    /// The backing reqwest client uses [`DEFAULT_HTTP_TIMEOUT`] and the
    /// transport uses [`DEFAULT_SEND_DEADLINE`]. To *raise* the request timeout
    /// past the default minute (e.g. for long builds / codegen), use
    /// [`StreamableHttpTransport::builder_with_timeout`] — calling
    /// [`StreamableHttpTransport::with_request_timeout`] on a builder produced
    /// here only relaxes the send deadline and cannot lift the client's own
    /// [`DEFAULT_HTTP_TIMEOUT`] (see its docs).
    ///
    /// Wrap the result in `Arc` before handing it to `McpClient::new`:
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use agent_sdk::mcp::{McpAuth, StreamableHttpTransport};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let transport = Arc::new(
    ///     StreamableHttpTransport::builder("https://example.com/mcp", McpAuth::None)?
    ///         .with_header("X-Tenant-Id", "acme"),
    /// );
    /// # let _ = transport;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying HTTP client cannot be built.
    pub fn builder(endpoint: impl Into<String>, auth: McpAuth) -> Result<Self> {
        let poster = ReqwestPoster::new(endpoint)?;
        Ok(Self::with_poster_owned(Arc::new(poster), auth))
    }

    /// Create an un-wrapped transport over real HTTP with a custom request
    /// timeout, for further builder-style configuration before wrapping in
    /// `Arc`.
    ///
    /// Sets *both* the backing reqwest client's request timeout *and* the
    /// transport-level send deadline to `request_timeout`. This is the path to
    /// use when **raising** the timeout past [`DEFAULT_HTTP_TIMEOUT`]: building
    /// the client with the higher timeout is the only way a long-running tool
    /// call (build, codegen) can run past the default minute — chaining
    /// [`StreamableHttpTransport::with_request_timeout`] onto a plain
    /// [`StreamableHttpTransport::builder`] cannot, because the underlying
    /// reqwest client was already built with [`DEFAULT_HTTP_TIMEOUT`].
    ///
    /// Mirrors [`StreamableHttpTransport::with_timeout`] but returns an
    /// un-wrapped transport so callers can chain
    /// [`StreamableHttpTransport::with_header`] before wrapping in `Arc`:
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use std::time::Duration;
    /// use agent_sdk::mcp::{McpAuth, StreamableHttpTransport};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let transport = Arc::new(
    ///     StreamableHttpTransport::builder_with_timeout(
    ///         "https://example.com/mcp",
    ///         McpAuth::None,
    ///         Duration::from_secs(300),
    ///     )?
    ///     .with_header("X-Tenant-Id", "acme"),
    /// );
    /// # let _ = transport;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying HTTP client cannot be built.
    pub fn builder_with_timeout(
        endpoint: impl Into<String>,
        auth: McpAuth,
        request_timeout: Duration,
    ) -> Result<Self> {
        let poster = ReqwestPoster::with_timeout(endpoint, request_timeout)?;
        Ok(Self::with_poster_owned(Arc::new(poster), auth).with_request_timeout(request_timeout))
    }

    /// Create an un-wrapped transport backed by a custom [`HttpPoster`], for
    /// further builder-style configuration before wrapping in `Arc`.
    #[must_use]
    pub fn with_poster_owned(poster: Arc<dyn HttpPoster>, auth: McpAuth) -> Self {
        Self {
            poster,
            auth,
            extra_headers: Vec::new(),
            next_id: AtomicU64::new(1),
            session_id: RwLock::new(None),
            protocol_version: RwLock::new(None),
            send_deadline: DEFAULT_SEND_DEADLINE,
        }
    }

    /// Add a static custom header sent on every request (e.g. a tenant id).
    ///
    /// Call this on an un-wrapped transport from [`StreamableHttpTransport::builder`]
    /// (or [`StreamableHttpTransport::with_poster_owned`]) before wrapping it in
    /// `Arc`.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Set the overall per-request send deadline (default
    /// [`DEFAULT_SEND_DEADLINE`]).
    ///
    /// This bounds every [`McpTransport::send`] / `send_notification` call
    /// regardless of the underlying [`HttpPoster`]'s own timeout, so a custom
    /// poster (or a [`ReqwestPoster`] whose client timeout is longer) still has
    /// a guaranteed cancellation path. Call it on an un-wrapped transport from
    /// [`StreamableHttpTransport::builder`] or
    /// [`StreamableHttpTransport::with_poster_owned`] before wrapping in `Arc`.
    ///
    /// # This only *lowers* the effective timeout, never raises it
    ///
    /// The effective per-request bound is the **minimum** of this send deadline
    /// and the backing [`HttpPoster`]'s own timeout. For a [`ReqwestPoster`]
    /// built via [`StreamableHttpTransport::builder`] /
    /// [`ReqwestPoster::new`], that client timeout is [`DEFAULT_HTTP_TIMEOUT`]
    /// (60s), so:
    ///
    /// * **Lowering** works: `with_request_timeout(Duration::from_secs(5))`
    ///   trips the send deadline at 5s, well before the client's 60s.
    /// * **Raising does *not* work here:**
    ///   `with_request_timeout(Duration::from_secs(300))` leaves the send
    ///   deadline at 300s but the client still aborts the request at its own
    ///   60s [`DEFAULT_HTTP_TIMEOUT`]. To genuinely raise the timeout, build the
    ///   transport with [`StreamableHttpTransport::builder_with_timeout`] (or
    ///   [`StreamableHttpTransport::with_timeout`]), which configures both the
    ///   reqwest client timeout and this send deadline together.
    #[must_use]
    pub const fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.send_deadline = request_timeout;
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

    /// Overall send deadline configured on this transport.
    ///
    /// Test-only accessor used to assert the builder stores the caller's value
    /// (or the documented default when none is given).
    #[cfg(test)]
    const fn send_deadline(&self) -> Duration {
        self.send_deadline
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

/// Compare two JSON-RPC ids, tolerating a server that echoes a numeric id as a
/// string (or vice-versa) — but nothing looser.
fn ids_match(a: &RequestId, b: &RequestId) -> bool {
    match (a, b) {
        (RequestId::Number(x), RequestId::Number(y)) => x == y,
        (RequestId::String(x), RequestId::String(y)) => x == y,
        (RequestId::Number(n), RequestId::String(s))
        | (RequestId::String(s), RequestId::Number(n)) => s.parse::<u64>().ok() == Some(*n),
    }
}

/// Extract the matching JSON-RPC response from an SSE body.
///
/// Returns the first `data:` payload that parses as a [`JsonRpcResponse`] whose
/// `id` matches `id`. Server-initiated requests/notifications carried on the
/// same stream (sampling, roots/list, elicitation — all of which include a
/// `method` field) are skipped, and a message whose id does not match is *not*
/// substituted as a fallback: if nothing matches, this is an error rather than
/// silently returning the wrong message as the reply.
fn parse_sse_response(body: &str, id: &RequestId) -> Result<JsonRpcResponse> {
    let mut data_buf = String::new();

    let try_match = |data: &mut String| -> Option<JsonRpcResponse> {
        if data.is_empty() {
            return None;
        }
        let raw = std::mem::take(data);
        let trimmed = raw.trim();
        // Skip server-initiated requests/notifications: those carry a `method`
        // and are not a reply to our request, even though they deserialize into
        // `JsonRpcResponse` (result/error both optional, unknown fields ignored).
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
            && value.get("method").is_some()
        {
            return None;
        }
        if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(trimmed)
            && ids_match(&resp.id, id)
        {
            return Some(resp);
        }
        None
    };

    for line in body.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            // Event boundary: attempt to resolve the accumulated data block.
            if let Some(resp) = try_match(&mut data_buf) {
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
    if let Some(resp) = try_match(&mut data_buf) {
        return Ok(resp);
    }

    bail!("SSE stream contained no JSON-RPC response matching the request id")
}

#[async_trait]
impl McpTransport for StreamableHttpTransport {
    async fn send(&self, mut request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let id = self.next_request_id();
        request.id = RequestId::Number(id);
        let request_id = request.id.clone();

        let body = serde_json::to_string(&request).context("failed to serialize MCP request")?;
        let http_request = self.build_http_request(body).await;
        // Overall deadline so a hung/keep-alive server can never wedge a turn.
        let reply = tokio::time::timeout(self.send_deadline, self.poster.post(http_request))
            .await
            .context("MCP HTTP request timed out")??;
        self.capture_session_id(&reply).await;

        let response = parse_reply(&reply, &request_id)?;

        if let Some(ref error) = response.error {
            bail!("JSON-RPC error {}: {}", error.code, error.message);
        }
        Ok(response)
    }

    async fn send_notification(&self, mut request: JsonRpcRequest) -> Result<()> {
        // Advance the shared id counter so request ids stay monotonic across the
        // connection, but strip the id on the wire: JSON-RPC 2.0 / MCP
        // notifications must not carry one.
        let id = self.next_request_id();
        request.id = RequestId::Number(id);
        let body = notification_body(&request)?;
        let http_request = self.build_http_request(body).await;
        let reply = tokio::time::timeout(self.send_deadline, self.poster.post(http_request))
            .await
            .context("MCP HTTP request timed out")??;
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
    /// Request timeout the backing client was built with. `reqwest::Client`
    /// does not expose its configured timeout, so we record it to let tests
    /// assert that a *raised* timeout actually reaches the client (not just the
    /// transport's send deadline).
    configured_timeout: Option<Duration>,
}

impl ReqwestPoster {
    /// Build a reqwest-backed poster for `endpoint`.
    ///
    /// The client is given a default request timeout
    /// (`DEFAULT_HTTP_TIMEOUT`) so a slow, hung, or keep-alive SSE server
    /// cannot block a request forever. Use [`ReqwestPoster::with_client`] to
    /// supply a client with different settings.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be constructed.
    pub fn new(endpoint: impl Into<String>) -> Result<Self> {
        Self::with_timeout(endpoint, DEFAULT_HTTP_TIMEOUT)
    }

    /// Build a reqwest-backed poster for `endpoint` with a custom request
    /// timeout instead of [`DEFAULT_HTTP_TIMEOUT`].
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be constructed.
    pub fn with_timeout(endpoint: impl Into<String>, timeout: Duration) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .context("failed to build MCP HTTP client")?;
        Ok(Self {
            client,
            endpoint: endpoint.into(),
            configured_timeout: Some(timeout),
        })
    }

    /// Build a poster from a caller-supplied `reqwest::Client`.
    #[must_use]
    pub fn with_client(client: reqwest::Client, endpoint: impl Into<String>) -> Self {
        Self {
            client,
            endpoint: endpoint.into(),
            configured_timeout: None,
        }
    }

    /// Request timeout the backing reqwest client was built with, if known.
    ///
    /// Returns `None` for posters built from a caller-supplied client via
    /// [`ReqwestPoster::with_client`], since `reqwest::Client` does not expose
    /// its own configured timeout. Useful to confirm a *raised* request timeout
    /// actually reached the client rather than only the transport's send
    /// deadline.
    #[must_use]
    pub const fn configured_timeout(&self) -> Option<Duration> {
        self.configured_timeout
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

        let mut response = builder
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

        // Read the body incrementally with a hard cap so an endless SSE stream
        // (kept open with keep-alive comments) cannot grow memory without bound.
        let mut body_bytes: Vec<u8> = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .context("failed to read MCP HTTP response body")?
        {
            if body_bytes.len() + chunk.len() > MAX_RESPONSE_BODY_BYTES {
                bail!("MCP HTTP response body exceeds {MAX_RESPONSE_BODY_BYTES} bytes");
            }
            body_bytes.extend_from_slice(&chunk);
        }
        let body = String::from_utf8_lossy(&body_bytes).into_owned();

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

    /// The transport defaults to the documented send deadline, and
    /// `with_request_timeout` overrides it.
    #[test]
    fn send_deadline_defaults_and_overrides() {
        let poster = Arc::new(CapturingPoster {
            last_body: std::sync::Mutex::new(None),
        });
        let default = StreamableHttpTransport::with_poster_owned(poster.clone(), McpAuth::None);
        assert_eq!(default.send_deadline(), DEFAULT_SEND_DEADLINE);

        let custom = StreamableHttpTransport::with_poster_owned(poster, McpAuth::None)
            .with_request_timeout(Duration::from_millis(250));
        assert_eq!(custom.send_deadline(), Duration::from_millis(250));
    }

    /// `ReqwestPoster::with_timeout` and `StreamableHttpTransport::with_timeout`
    /// must build successfully and the transport must record the configured
    /// deadline.
    #[test]
    fn reqwest_with_timeout_builds_and_transport_records_deadline() -> Result<()> {
        ReqwestPoster::with_timeout("https://example.com/mcp", Duration::from_secs(5))?;
        let transport = StreamableHttpTransport::with_timeout(
            "https://example.com/mcp",
            McpAuth::None,
            Duration::from_secs(5),
        )?;
        assert_eq!(transport.send_deadline(), Duration::from_secs(5));
        Ok(())
    }

    /// Regression test for the builder-path footgun: a *raised* request timeout
    /// (300s, well past the 60s [`DEFAULT_HTTP_TIMEOUT`]) must reach BOTH the
    /// backing reqwest client and the transport send deadline. Previously,
    /// raising the timeout via the builder silently left the client capped at
    /// [`DEFAULT_HTTP_TIMEOUT`], so only lowering ever took effect.
    #[test]
    fn builder_with_timeout_raises_client_timeout_and_send_deadline() -> Result<()> {
        let raised = Duration::from_mins(5);

        // The backing poster's client must carry the raised timeout, not the
        // 60s default — this is the bit that was silently ignored before.
        let poster = ReqwestPoster::with_timeout("https://example.com/mcp", raised)?;
        assert_eq!(
            poster.configured_timeout(),
            Some(raised),
            "raised timeout must reach the reqwest client, not stay at DEFAULT_HTTP_TIMEOUT"
        );
        assert_ne!(
            poster.configured_timeout(),
            Some(DEFAULT_HTTP_TIMEOUT),
            "client must not stay capped at the default minute when the caller raised it"
        );

        // And the transport built via the builder path must also carry the
        // raised send deadline (so neither bound silently caps the request).
        let transport = StreamableHttpTransport::builder_with_timeout(
            "https://example.com/mcp",
            McpAuth::None,
            raised,
        )?;
        assert_eq!(transport.send_deadline(), raised);

        Ok(())
    }

    /// A poster that stalls forever must trip the configured send deadline
    /// quickly rather than blocking for the full default minute.
    #[tokio::test]
    async fn configured_send_deadline_fails_fast() -> Result<()> {
        struct StallingPoster;

        #[async_trait]
        impl HttpPoster for StallingPoster {
            async fn post(&self, _request: HttpRequest) -> Result<HttpReply> {
                // Far longer than the configured deadline; the outer timeout
                // cancels this future well before it resolves.
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok(HttpReply::json("{}"))
            }
        }

        let transport = Arc::new(
            StreamableHttpTransport::with_poster_owned(Arc::new(StallingPoster), McpAuth::None)
                .with_request_timeout(Duration::from_millis(50)),
        );

        let started = std::time::Instant::now();
        let result = transport.send(JsonRpcRequest::new("ping", None, 0)).await;
        assert!(
            result.is_err(),
            "a stalled server must trip the configured send deadline"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "must fail fast at the configured deadline, not wait the default minute (elapsed {:?})",
            started.elapsed(),
        );
        Ok(())
    }

    #[test]
    fn bearer_auth_header_value() {
        assert_eq!(McpAuth::None.header_value(), None);
        assert_eq!(
            McpAuth::Bearer("tok".to_string()).header_value().as_deref(),
            Some("Bearer tok"),
        );
    }

    /// Regression test for finding 8: an SSE stream that contains no message
    /// matching the request id must error, not return a non-matching message as
    /// a fallback (which previously masked server-initiated messages as the
    /// reply).
    #[test]
    fn parse_sse_no_matching_id_is_error() {
        let body = format!(
            "data: {}\n\n",
            ok_response(99, &serde_json::json!({"x": 1}))
        );
        let reply = HttpReply::event_stream(body);
        let result = parse_reply(&reply, &RequestId::Number(3));
        assert!(
            result.is_err(),
            "a stream with no matching id must error rather than return a fallback"
        );
    }

    /// Regression test for finding 8: a server-initiated request carried on the
    /// stream (it has a `method` and even shares our id) must be skipped, and
    /// the real reply returned.
    #[test]
    fn parse_sse_skips_server_request_with_method() -> Result<()> {
        let server_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "sampling/createMessage",
            "params": {},
        })
        .to_string();
        let body = format!(
            "data: {server_request}\n\ndata: {}\n\n",
            ok_response(3, &serde_json::json!({"answer": 42})),
        );
        let reply = HttpReply::event_stream(body);
        let resp = parse_reply(&reply, &RequestId::Number(3))?;
        assert_eq!(resp.id, RequestId::Number(3));
        assert!(
            resp.result().is_some(),
            "must return the real reply, not the server request"
        );
        Ok(())
    }

    #[test]
    fn ids_match_coerces_numeric_string() {
        assert!(ids_match(&RequestId::Number(5), &RequestId::Number(5)));
        assert!(ids_match(
            &RequestId::Number(5),
            &RequestId::String("5".to_string())
        ));
        assert!(ids_match(
            &RequestId::String("5".to_string()),
            &RequestId::Number(5)
        ));
        assert!(!ids_match(
            &RequestId::Number(5),
            &RequestId::String("six".to_string())
        ));
        assert!(!ids_match(&RequestId::Number(5), &RequestId::Number(6)));
    }

    /// Poster that records the most recent body it was asked to POST.
    struct CapturingPoster {
        last_body: std::sync::Mutex<Option<String>>,
    }

    #[async_trait]
    impl HttpPoster for CapturingPoster {
        async fn post(&self, request: HttpRequest) -> Result<HttpReply> {
            *self
                .last_body
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(request.body);
            Ok(HttpReply::json(ok_response(1, &serde_json::json!({}))))
        }
    }

    /// Regression test for finding 13: HTTP notifications must be serialized
    /// without an `id`.
    #[tokio::test]
    async fn send_notification_omits_id() -> Result<()> {
        let poster = Arc::new(CapturingPoster {
            last_body: std::sync::Mutex::new(None),
        });
        let transport = StreamableHttpTransport::with_poster(poster.clone(), McpAuth::None);

        transport
            .send_notification(JsonRpcRequest::new("notifications/initialized", None, 0))
            .await?;

        let body = poster
            .last_body
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .context("no body captured")?;
        let value: serde_json::Value = serde_json::from_str(&body)?;
        assert!(
            value.get("id").is_none(),
            "notification must not carry an id, got: {body}"
        );
        assert_eq!(
            value.get("method").and_then(serde_json::Value::as_str),
            Some("notifications/initialized")
        );
        Ok(())
    }

    /// `with_header` must be reachable via the builder and the header must be
    /// forwarded on requests (finding 7).
    #[tokio::test]
    async fn builder_with_header_is_forwarded() -> Result<()> {
        struct HeaderCapturingPoster {
            headers: std::sync::Mutex<Vec<(String, String)>>,
        }

        #[async_trait]
        impl HttpPoster for HeaderCapturingPoster {
            async fn post(&self, request: HttpRequest) -> Result<HttpReply> {
                *self
                    .headers
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = request.extra_headers;
                Ok(HttpReply::json(ok_response(1, &serde_json::json!({}))))
            }
        }

        let poster = Arc::new(HeaderCapturingPoster {
            headers: std::sync::Mutex::new(Vec::new()),
        });
        let transport = Arc::new(
            StreamableHttpTransport::with_poster_owned(poster.clone(), McpAuth::None)
                .with_header("X-Tenant-Id", "acme"),
        );

        transport.send(JsonRpcRequest::new("ping", None, 0)).await?;

        let headers = poster
            .headers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert!(
            headers
                .iter()
                .any(|(k, v)| k == "X-Tenant-Id" && v == "acme"),
            "custom header set via builder must be forwarded, got: {headers:?}"
        );
        Ok(())
    }
}
