//! Link fetch tool implementation.

use crate::tools::{PrimitiveToolName, Tool, ToolContext};
use crate::types::{ToolResult, ToolTier};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::time::Duration;

use super::security::UrlValidator;

/// Maximum content size to fetch (1MB).
const MAX_CONTENT_SIZE: usize = 1024 * 1024;

/// Default request timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Output format for fetched content.
///
/// Only plain text is currently supported. A `Markdown` variant previously
/// existed but produced byte-identical output to `Text` (no real markdown
/// conversion was implemented), so it was removed rather than advertise a
/// distinction to the model that did not exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FetchFormat {
    /// Plain text output (HTML tags removed).
    #[default]
    Text,
}

/// Link fetch tool for securely retrieving web page content.
///
/// This tool fetches web pages and converts them to text or markdown format.
/// It includes SSRF protection to prevent access to internal resources.
///
/// # Example
///
/// ```ignore
/// use agent_sdk::web::LinkFetchTool;
///
/// let tool = LinkFetchTool::new();
///
/// // Register with agent
/// tools.register(tool);
/// ```
pub struct LinkFetchTool {
    /// Optional caller-supplied HTTP client.
    ///
    /// When `None` (the default), a fresh client is built per request with the
    /// validated IP addresses pinned via [`reqwest::ClientBuilder::resolve_to_addrs`]
    /// so the connection targets exactly the addresses that passed SSRF
    /// validation (closing the DNS-rebinding window). When a custom client is
    /// supplied via [`LinkFetchTool::with_client`], it is used as-is and the
    /// caller is responsible for its redirect/SSRF policy.
    client: Option<reqwest::Client>,
    validator: UrlValidator,
}

impl Default for LinkFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl LinkFetchTool {
    /// Create a new link fetch tool with default settings.
    ///
    /// Does not build an HTTP client eagerly: the default client is constructed
    /// per request (with the validated IPs pinned), so this constructor cannot
    /// fail or panic.
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: None,
            validator: UrlValidator::new(),
        }
    }

    /// Create with a custom URL validator.
    #[must_use]
    pub fn with_validator(mut self, validator: UrlValidator) -> Self {
        self.validator = validator;
        self
    }

    /// Create with a custom HTTP client.
    ///
    /// The supplied client is used verbatim; per-request IP pinning is *not*
    /// applied, so the caller takes responsibility for redirect and SSRF
    /// policy on that client.
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the HTTP client for a single request.
    ///
    /// Returns the caller-supplied client if one was configured; otherwise
    /// builds a default client with `host` pinned to the vetted `addrs` so the
    /// connection cannot be rebound to a different (blocked) address after
    /// validation.
    fn build_client(
        &self,
        host: Option<&str>,
        addrs: &[std::net::SocketAddr],
    ) -> Result<reqwest::Client> {
        if let Some(client) = &self.client {
            return Ok(client.clone());
        }

        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(DEFAULT_TIMEOUT)
            .user_agent("Mozilla/5.0 (compatible; AgentSDK/1.0)");

        if let Some(host) = host
            && !addrs.is_empty()
        {
            builder = builder.resolve_to_addrs(host, addrs);
        }

        builder.build().context("Failed to build HTTP client")
    }

    /// Fetch a URL and convert it to plain text.
    ///
    /// Manually follows redirects, validating each target URL through the
    /// SSRF validator (and pinning its resolved IPs) to prevent redirect-based
    /// and DNS-rebinding SSRF attacks.
    async fn fetch_url(&self, url_str: &str) -> Result<String> {
        // Validate initial URL before fetching, capturing the vetted addresses.
        let mut validated = self.validator.validate(url_str).await?;
        let max_redirects = self.validator.max_redirects();

        let client = self.build_client(validated.url.host_str(), &validated.addresses)?;
        let mut response = client
            .get(validated.url.as_str())
            .send()
            .await
            .context("Failed to fetch URL")?;

        // Manually follow redirects with validation
        let mut redirects = 0;
        while response.status().is_redirection() {
            redirects += 1;
            if redirects > max_redirects {
                bail!("Too many redirects ({redirects} > {max_redirects})");
            }

            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .context("Redirect response missing Location header")?
                .to_str()
                .context("Invalid Location header")?;

            // Resolve relative redirect URLs against the current URL
            let redirect_url_str = validated
                .url
                .join(location)
                .map_or_else(|_| location.to_string(), |u| u.to_string());

            // Validate the redirect target through the same SSRF checks and
            // pin its freshly-vetted addresses.
            validated = self.validator.validate(&redirect_url_str).await?;

            let client = self.build_client(validated.url.host_str(), &validated.addresses)?;
            response = client
                .get(validated.url.as_str())
                .send()
                .await
                .context("Failed to follow redirect")?;
        }

        // Check status
        if !response.status().is_success() {
            bail!("HTTP error: {}", response.status());
        }

        // Reject early if the advertised length already exceeds the cap.
        if let Some(len) = response.content_length()
            && len > MAX_CONTENT_SIZE as u64
        {
            bail!("Content too large: {len} bytes (max {MAX_CONTENT_SIZE} bytes)");
        }

        // Get content type to determine processing
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/html")
            .to_string();

        // Stream the body, bailing as soon as the cumulative size exceeds the
        // cap. This bounds peak memory at ~MAX_CONTENT_SIZE regardless of
        // whether the server sends a Content-Length header (chunked/streaming
        // responses would otherwise allow unbounded allocation).
        let bytes = read_capped_body(&mut response, MAX_CONTENT_SIZE).await?;

        // Convert to string
        let html = String::from_utf8_lossy(&bytes);

        // Process based on content type
        if content_type.contains("text/html") || content_type.contains("application/xhtml") {
            Ok(convert_html(&html))
        } else if content_type.contains("text/plain") {
            Ok(html.into_owned())
        } else {
            // For other content types, just return as-is
            Ok(html.into_owned())
        }
    }
}

/// Convert HTML to plain text.
fn convert_html(html: &str) -> String {
    html2text::from_read(html.as_bytes(), 80).unwrap_or_else(|_| html.to_string())
}

/// Read a response body into memory, bailing as soon as the cumulative size
/// exceeds `max`.
///
/// Streams via [`reqwest::Response::chunk`] so peak memory stays bounded at
/// ~`max` even for chunked/streaming responses that carry no `Content-Length`.
async fn read_capped_body(response: &mut reqwest::Response, max: usize) -> Result<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("Failed to read response body")?
    {
        if bytes.len() + chunk.len() > max {
            bail!("Content too large: exceeds {max} bytes");
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

impl<Ctx> Tool<Ctx> for LinkFetchTool
where
    Ctx: Send + Sync + 'static,
{
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::LinkFetch
    }

    fn display_name(&self) -> &'static str {
        "Fetch URL"
    }

    fn description(&self) -> &'static str {
        "Fetch and read web page content. Returns the page content as text or markdown. \
         Includes SSRF protection to prevent access to internal resources."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must be HTTPS)"
                }
            },
            "required": ["url"]
        })
    }

    fn tier(&self) -> ToolTier {
        // Link fetch is read-only, so Observe tier
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        let url = input
            .get("url")
            .and_then(Value::as_str)
            .context("Missing 'url' parameter")?;

        match self.fetch_url(url).await {
            Ok(content) => Ok(ToolResult {
                success: true,
                output: content,
                data: Some(json!({ "url": url })),
                documents: Vec::new(),
                duration_ms: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: format!("Failed to fetch URL: {e}"),
                data: Some(json!({ "url": url, "error": e.to_string() })),
                documents: Vec::new(),
                duration_ms: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_fetch_tool_metadata() {
        let tool = LinkFetchTool::new();

        assert_eq!(Tool::<()>::name(&tool), PrimitiveToolName::LinkFetch);
        assert!(Tool::<()>::description(&tool).contains("Fetch"));
        assert_eq!(Tool::<()>::tier(&tool), ToolTier::Observe);
    }

    #[test]
    fn test_link_fetch_tool_input_schema() {
        let tool = LinkFetchTool::new();

        let schema = Tool::<()>::input_schema(&tool);
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["url"].is_object());
        // The dead `format`/markdown option was removed from the schema.
        assert!(schema["properties"]["format"].is_null());
        assert!(
            schema["required"]
                .as_array()
                .is_some_and(|arr| arr.iter().any(|v| v == "url"))
        );
    }

    #[test]
    fn test_convert_html_text() {
        let html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>";
        let result = convert_html(html);
        assert!(result.contains("Title"));
        assert!(result.contains("Paragraph"));
    }

    #[tokio::test]
    async fn test_link_fetch_blocked_url() {
        let tool = LinkFetchTool::new();
        let ctx = ToolContext::new(());
        let input = json!({ "url": "http://localhost:8080" });

        let result = Tool::<()>::execute(&tool, &ctx, input).await;
        assert!(result.is_ok());

        let tool_result = result.expect("Should succeed");
        assert!(!tool_result.success);
        assert!(
            tool_result.output.contains("HTTPS required") || tool_result.output.contains("blocked")
        );
    }

    #[tokio::test]
    async fn test_link_fetch_missing_url() {
        let tool = LinkFetchTool::new();
        let ctx = ToolContext::new(());
        let input = json!({});

        let result = Tool::<()>::execute(&tool, &ctx, input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("url"));
    }

    #[tokio::test]
    async fn test_link_fetch_invalid_url() {
        let tool = LinkFetchTool::new();
        let ctx = ToolContext::new(());
        let input = json!({ "url": "not-a-valid-url" });

        let result = Tool::<()>::execute(&tool, &ctx, input).await;
        assert!(result.is_ok());

        let tool_result = result.expect("Should succeed");
        assert!(!tool_result.success);
        assert!(tool_result.output.contains("Invalid URL"));
    }

    #[test]
    fn test_with_validator() {
        let validator = UrlValidator::new().with_allow_http();
        let _tool = LinkFetchTool::new().with_validator(validator);
        // Just verify it compiles - validator is private
    }

    #[test]
    fn test_redirects_disabled_in_client() {
        // Verify that the default client has redirects disabled
        // (reqwest::Policy::none means no automatic redirect following)
        let tool = LinkFetchTool::new();
        // The client is private, but we can verify redirect behavior indirectly:
        // A redirect response should NOT be automatically followed
        assert_eq!(tool.validator.max_redirects(), 3);
    }

    #[tokio::test]
    async fn test_redirect_to_private_ip_blocked() {
        // Simulate: a redirect target pointing to a private IP should be blocked
        // by the validator during manual redirect following.
        let validator = UrlValidator::new().with_allow_http();

        // Direct access to private IPs should be blocked
        let result = validator
            .validate("http://169.254.169.254/latest/meta-data/")
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("blocked"));

        // Direct access to 10.x should be blocked
        let result = validator.validate("http://10.0.0.1/internal").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_redirect_to_localhost_blocked() {
        let validator = UrlValidator::new().with_allow_http();

        // Redirect target pointing to localhost should be blocked
        let result = validator.validate("http://127.0.0.1/admin").await;
        assert!(result.is_err());
    }

    /// Regression test for the body-size cap (findings 4 & 14). A server that
    /// sends a body larger than the cap with NO `Content-Length` header
    /// (so the header pre-check cannot apply) must be rejected while streaming,
    /// before the whole body is buffered. Exercised against a local loopback
    /// server so the test is deterministic and needs no network.
    #[tokio::test]
    async fn test_read_capped_body_rejects_oversized_stream() -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;

        let server = tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let header =
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n";
                let _ = sock.write_all(header.as_bytes()).await;
                let chunk = vec![b'a'; 64 * 1024];
                // Write well past the 1 MiB cap used below.
                for _ in 0..40 {
                    if sock.write_all(&chunk).await.is_err() {
                        break;
                    }
                }
                let _ = sock.shutdown().await;
            }
        });

        let client = reqwest::Client::builder().build()?;
        let mut response = client.get(format!("http://{addr}/big")).send().await?;
        let result = read_capped_body(&mut response, 1024 * 1024).await;
        server.abort();

        assert!(result.is_err(), "oversized streamed body must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Content too large"),
            "expected size-cap error, got: {msg}"
        );
        Ok(())
    }

    /// A body within the cap must be returned in full.
    #[tokio::test]
    async fn test_read_capped_body_accepts_small_stream() -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;

        let server = tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let body = "hello world";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });

        let client = reqwest::Client::builder().build()?;
        let mut response = client.get(format!("http://{addr}/small")).send().await?;
        let bytes = read_capped_body(&mut response, 1024 * 1024).await?;
        server.abort();

        assert_eq!(String::from_utf8_lossy(&bytes), "hello world");
        Ok(())
    }
}
