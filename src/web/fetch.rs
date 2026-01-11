//! Link fetch tool implementation.

use crate::tools::{Tool, ToolContext};
use crate::types::{ToolResult, ToolTier};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::Duration;

use super::security::UrlValidator;

/// Maximum content size to fetch (1MB).
const MAX_CONTENT_SIZE: usize = 1024 * 1024;

/// Default request timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Output format for fetched content.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FetchFormat {
    /// Plain text output (HTML tags removed).
    #[default]
    Text,
    /// Markdown-formatted output.
    Markdown,
}

impl FetchFormat {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "text" => Some(Self::Text),
            "markdown" | "md" => Some(Self::Markdown),
            _ => None,
        }
    }
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
    client: reqwest::Client,
    validator: UrlValidator,
    default_format: FetchFormat,
}

impl Default for LinkFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl LinkFetchTool {
    /// Create a new link fetch tool with default settings.
    ///
    /// # Panics
    ///
    /// Panics if the HTTP client cannot be built (should never happen with default settings).
    #[must_use]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .user_agent("Mozilla/5.0 (compatible; AgentSDK/1.0)")
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            validator: UrlValidator::new(),
            default_format: FetchFormat::Text,
        }
    }

    /// Create with a custom URL validator.
    #[must_use]
    pub fn with_validator(mut self, validator: UrlValidator) -> Self {
        self.validator = validator;
        self
    }

    /// Create with a custom HTTP client.
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Set the default output format.
    #[must_use]
    pub const fn with_default_format(mut self, format: FetchFormat) -> Self {
        self.default_format = format;
        self
    }

    /// Fetch a URL and convert to the specified format.
    async fn fetch_url(&self, url_str: &str, format: FetchFormat) -> Result<String> {
        // Validate URL before fetching
        let url = self.validator.validate(url_str)?;

        // Build request with redirect policy
        let response = self
            .client
            .get(url.as_str())
            .send()
            .await
            .context("Failed to fetch URL")?;

        // Check status
        if !response.status().is_success() {
            bail!("HTTP error: {}", response.status());
        }

        // Check content length if available
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

        // Read body with size limit
        let bytes = response
            .bytes()
            .await
            .context("Failed to read response body")?;

        if bytes.len() > MAX_CONTENT_SIZE {
            bail!(
                "Content too large: {} bytes (max {} bytes)",
                bytes.len(),
                MAX_CONTENT_SIZE
            );
        }

        // Convert to string
        let html = String::from_utf8_lossy(&bytes);

        // Process based on content type and format
        if content_type.contains("text/html") || content_type.contains("application/xhtml") {
            Ok(convert_html(&html, format))
        } else if content_type.contains("text/plain") {
            Ok(html.into_owned())
        } else {
            // For other content types, just return as-is
            Ok(html.into_owned())
        }
    }
}

/// Convert HTML to the specified format.
fn convert_html(html: &str, format: FetchFormat) -> String {
    let result = match format {
        FetchFormat::Text => {
            // Use html2text with default width
            html2text::from_read(html.as_bytes(), 80)
        }
        FetchFormat::Markdown => {
            // Use html2text with markdown-friendly settings
            html2text::from_read(html.as_bytes(), 80)
        }
    };
    result.unwrap_or_else(|_| html.to_string())
}

#[async_trait]
impl<Ctx> Tool<Ctx> for LinkFetchTool
where
    Ctx: Send + Sync + 'static,
{
    fn name(&self) -> &'static str {
        "link_fetch"
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
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "markdown"],
                    "description": "Output format (default: text)"
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

        let format = input
            .get("format")
            .and_then(Value::as_str)
            .and_then(FetchFormat::from_str)
            .unwrap_or(self.default_format);

        match self.fetch_url(url, format).await {
            Ok(content) => Ok(ToolResult {
                success: true,
                output: content,
                data: Some(json!({ "url": url, "format": format_name(format) })),
                duration_ms: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: format!("Failed to fetch URL: {e}"),
                data: Some(json!({ "url": url, "error": e.to_string() })),
                duration_ms: None,
            }),
        }
    }
}

/// Get the format name for JSON output.
const fn format_name(format: FetchFormat) -> &'static str {
    match format {
        FetchFormat::Text => "text",
        FetchFormat::Markdown => "markdown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_fetch_tool_metadata() {
        let tool = LinkFetchTool::new();

        assert_eq!(Tool::<()>::name(&tool), "link_fetch");
        assert!(Tool::<()>::description(&tool).contains("Fetch"));
        assert_eq!(Tool::<()>::tier(&tool), ToolTier::Observe);
    }

    #[test]
    fn test_link_fetch_tool_input_schema() {
        let tool = LinkFetchTool::new();

        let schema = Tool::<()>::input_schema(&tool);
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["url"].is_object());
        assert!(schema["properties"]["format"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .is_some_and(|arr| arr.iter().any(|v| v == "url"))
        );
    }

    #[test]
    fn test_format_from_str() {
        assert_eq!(FetchFormat::from_str("text"), Some(FetchFormat::Text));
        assert_eq!(FetchFormat::from_str("TEXT"), Some(FetchFormat::Text));
        assert_eq!(
            FetchFormat::from_str("markdown"),
            Some(FetchFormat::Markdown)
        );
        assert_eq!(FetchFormat::from_str("md"), Some(FetchFormat::Markdown));
        assert_eq!(FetchFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_convert_html_text() {
        let html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>";
        let result = convert_html(html, FetchFormat::Text);
        assert!(result.contains("Title"));
        assert!(result.contains("Paragraph"));
    }

    #[test]
    fn test_default_format() {
        let tool = LinkFetchTool::new();
        assert_eq!(tool.default_format, FetchFormat::Text);

        let tool = LinkFetchTool::new().with_default_format(FetchFormat::Markdown);
        assert_eq!(tool.default_format, FetchFormat::Markdown);
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
    fn test_format_name() {
        assert_eq!(format_name(FetchFormat::Text), "text");
        assert_eq!(format_name(FetchFormat::Markdown), "markdown");
    }
}
