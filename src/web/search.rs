//! Web search tool implementation.

use crate::tools::{PrimitiveToolName, Tool, ToolContext};
use crate::types::{ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::fmt::Write;
use std::sync::Arc;

use super::provider::SearchProvider;

/// Web search tool that uses a configurable search provider.
///
/// This tool allows agents to search the web using any implementation
/// of the [`SearchProvider`] trait.
///
/// # Example
///
/// ```ignore
/// use agent_sdk::web::{WebSearchTool, BraveSearchProvider};
///
/// let provider = BraveSearchProvider::new("api-key");
/// let tool = WebSearchTool::new(provider);
///
/// // Register with agent
/// tools.register(tool);
/// ```
pub struct WebSearchTool<P: SearchProvider> {
    provider: Arc<P>,
    max_results: usize,
}

impl<P: SearchProvider> WebSearchTool<P> {
    /// Create a new web search tool with the given provider.
    #[must_use]
    pub fn new(provider: P) -> Self {
        Self {
            provider: Arc::new(provider),
            max_results: 10,
        }
    }

    /// Create a web search tool with a shared provider.
    #[must_use]
    pub const fn with_shared_provider(provider: Arc<P>) -> Self {
        Self {
            provider,
            max_results: 10,
        }
    }

    /// Set the default maximum number of results.
    #[must_use]
    pub const fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }
}

/// Format search results for display to the LLM.
fn format_search_results(query: &str, results: &[super::provider::SearchResult]) -> String {
    if results.is_empty() {
        return format!("No results found for: {query}");
    }

    let mut output = format!("Search results for: {query}\n\n");

    for (i, result) in results.iter().enumerate() {
        let _ = writeln!(output, "{}. {}", i + 1, result.title);
        let _ = writeln!(output, "   URL: {}", result.url);
        if !result.snippet.is_empty() {
            let _ = writeln!(output, "   {}", result.snippet);
        }
        if let Some(ref date) = result.published_date {
            let _ = writeln!(output, "   Published: {date}");
        }
        output.push('\n');
    }

    output
}

impl<Ctx, P> Tool<Ctx> for WebSearchTool<P>
where
    Ctx: Send + Sync + 'static,
    P: SearchProvider + 'static,
{
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::WebSearch
    }

    fn display_name(&self) -> &'static str {
        "Web Search"
    }

    fn description(&self) -> &'static str {
        "Search the web for current information. Returns titles, URLs, and snippets from search results."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10)"
                }
            },
            "required": ["query"]
        })
    }

    fn tier(&self) -> ToolTier {
        // Web search is read-only, so Observe tier
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .context("Missing 'query' parameter")?;

        let max_results = input
            .get("max_results")
            .and_then(Value::as_u64)
            .map_or(self.max_results, |n| {
                usize::try_from(n).unwrap_or(usize::MAX)
            });

        let response = self.provider.search(query, max_results).await?;

        let output = format_search_results(&response.query, &response.results);

        // Include structured data for programmatic access
        let data = serde_json::to_value(&response).ok();

        Ok(ToolResult {
            success: true,
            output,
            data,
            duration_ms: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Tool;
    use crate::web::provider::{SearchResponse, SearchResult};
    use async_trait::async_trait;

    // Mock provider for testing
    struct MockSearchProvider {
        results: Vec<SearchResult>,
    }

    impl MockSearchProvider {
        fn new(results: Vec<SearchResult>) -> Self {
            Self { results }
        }
    }

    #[async_trait]
    impl SearchProvider for MockSearchProvider {
        async fn search(&self, query: &str, max_results: usize) -> Result<SearchResponse> {
            Ok(SearchResponse {
                query: query.to_string(),
                results: self.results.iter().take(max_results).cloned().collect(),
                total_results: Some(self.results.len() as u64),
            })
        }

        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    #[test]
    fn test_web_search_tool_metadata() {
        let provider = MockSearchProvider::new(vec![]);
        let tool: WebSearchTool<MockSearchProvider> = WebSearchTool::new(provider);

        assert_eq!(Tool::<()>::name(&tool), PrimitiveToolName::WebSearch);
        assert!(Tool::<()>::description(&tool).contains("Search the web"));
        assert_eq!(Tool::<()>::tier(&tool), ToolTier::Observe);
    }

    #[test]
    fn test_web_search_tool_input_schema() {
        let provider = MockSearchProvider::new(vec![]);
        let tool: WebSearchTool<MockSearchProvider> = WebSearchTool::new(provider);

        let schema = Tool::<()>::input_schema(&tool);
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["query"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .is_some_and(|arr| arr.iter().any(|v| v == "query"))
        );
    }

    #[tokio::test]
    async fn test_web_search_tool_execute() -> Result<()> {
        let results = vec![
            SearchResult {
                title: "Rust Programming".into(),
                url: "https://rust-lang.org".into(),
                snippet: "A language empowering everyone".into(),
                published_date: None,
            },
            SearchResult {
                title: "Rust by Example".into(),
                url: "https://doc.rust-lang.org/rust-by-example".into(),
                snippet: "Learn Rust by example".into(),
                published_date: Some("2024-01-01".into()),
            },
        ];

        let provider = MockSearchProvider::new(results);
        let tool: WebSearchTool<MockSearchProvider> = WebSearchTool::new(provider);

        let ctx = ToolContext::new(());
        let input = json!({ "query": "rust programming" });

        let result = tool.execute(&ctx, input).await?;

        assert!(result.success);
        assert!(result.output.contains("Rust Programming"));
        assert!(result.output.contains("rust-lang.org"));
        assert!(result.data.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_web_search_tool_with_max_results() -> Result<()> {
        let results = vec![
            SearchResult {
                title: "Result 1".into(),
                url: "https://example.com/1".into(),
                snippet: "First".into(),
                published_date: None,
            },
            SearchResult {
                title: "Result 2".into(),
                url: "https://example.com/2".into(),
                snippet: "Second".into(),
                published_date: None,
            },
            SearchResult {
                title: "Result 3".into(),
                url: "https://example.com/3".into(),
                snippet: "Third".into(),
                published_date: None,
            },
        ];

        let provider = MockSearchProvider::new(results);
        let tool: WebSearchTool<MockSearchProvider> =
            WebSearchTool::new(provider).with_max_results(2);

        let ctx = ToolContext::new(());
        let input = json!({ "query": "test" });

        let result = tool.execute(&ctx, input).await?;

        assert!(result.success);
        // Should only show 2 results
        assert!(result.output.contains("Result 1"));
        assert!(result.output.contains("Result 2"));
        assert!(!result.output.contains("Result 3"));

        Ok(())
    }

    #[tokio::test]
    async fn test_web_search_tool_override_max_results() -> Result<()> {
        let results = vec![
            SearchResult {
                title: "Result 1".into(),
                url: "https://example.com/1".into(),
                snippet: "First".into(),
                published_date: None,
            },
            SearchResult {
                title: "Result 2".into(),
                url: "https://example.com/2".into(),
                snippet: "Second".into(),
                published_date: None,
            },
        ];

        let provider = MockSearchProvider::new(results);
        let tool: WebSearchTool<MockSearchProvider> = WebSearchTool::new(provider);

        let ctx = ToolContext::new(());
        // Override max_results in input
        let input = json!({ "query": "test", "max_results": 1 });

        let result = tool.execute(&ctx, input).await?;

        assert!(result.success);
        // Should only show 1 result
        assert!(result.output.contains("Result 1"));
        assert!(!result.output.contains("Result 2"));

        Ok(())
    }

    #[tokio::test]
    async fn test_web_search_tool_no_results() -> Result<()> {
        let provider = MockSearchProvider::new(vec![]);
        let tool: WebSearchTool<MockSearchProvider> = WebSearchTool::new(provider);

        let ctx = ToolContext::new(());
        let input = json!({ "query": "nonexistent query xyz" });

        let result = tool.execute(&ctx, input).await?;

        assert!(result.success);
        assert!(result.output.contains("No results found"));

        Ok(())
    }

    #[tokio::test]
    async fn test_web_search_tool_missing_query() {
        let provider = MockSearchProvider::new(vec![]);
        let tool: WebSearchTool<MockSearchProvider> = WebSearchTool::new(provider);

        let ctx = ToolContext::new(());
        let input = json!({});

        let result = tool.execute(&ctx, input).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("query"));
    }

    #[test]
    fn test_format_search_results_empty() {
        let output = format_search_results("test", &[]);
        assert!(output.contains("No results found"));
    }

    #[test]
    fn test_format_search_results_with_data() {
        let results = vec![
            SearchResult {
                title: "Title One".into(),
                url: "https://one.com".into(),
                snippet: "Snippet one".into(),
                published_date: Some("2024-01-15".into()),
            },
            SearchResult {
                title: "Title Two".into(),
                url: "https://two.com".into(),
                snippet: String::new(),
                published_date: None,
            },
        ];

        let output = format_search_results("query", &results);

        assert!(output.contains("Search results for: query"));
        assert!(output.contains("1. Title One"));
        assert!(output.contains("https://one.com"));
        assert!(output.contains("Snippet one"));
        assert!(output.contains("2024-01-15"));
        assert!(output.contains("2. Title Two"));
    }
}
