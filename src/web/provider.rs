//! Search provider trait and implementations.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A single search result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Title of the search result.
    pub title: String,
    /// URL of the result.
    pub url: String,
    /// Snippet/description of the result.
    pub snippet: String,
    /// Publication date, if available.
    pub published_date: Option<String>,
}

/// Response from a search query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    /// The original search query.
    pub query: String,
    /// List of search results.
    pub results: Vec<SearchResult>,
    /// Total number of results available (if known).
    pub total_results: Option<u64>,
}

/// Trait for search providers.
///
/// Implement this trait to add support for different search engines.
///
/// # Example
///
/// ```ignore
/// struct MySearchProvider { /* ... */ }
///
/// #[async_trait]
/// impl SearchProvider for MySearchProvider {
///     async fn search(&self, query: &str, max_results: usize) -> Result<SearchResponse> {
///         // Implementation
///     }
///
///     fn provider_name(&self) -> &'static str {
///         "my-provider"
///     }
/// }
/// ```
#[async_trait]
pub trait SearchProvider: Send + Sync {
    /// Execute a search query.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    /// * `max_results` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// Returns an error if the search request fails.
    async fn search(&self, query: &str, max_results: usize) -> Result<SearchResponse>;

    /// Get the provider name for logging/debugging.
    fn provider_name(&self) -> &'static str;
}

/// Brave Search API provider.
///
/// Uses the Brave Search API to perform web searches.
/// Requires a Brave Search API key from <https://brave.com/search/api/>
///
/// # Example
///
/// ```ignore
/// let provider = BraveSearchProvider::new("your-api-key");
/// let results = provider.search("rust programming", 10).await?;
/// ```
#[derive(Clone)]
pub struct BraveSearchProvider {
    client: reqwest::Client,
    api_key: String,
}

impl BraveSearchProvider {
    /// Create a new Brave Search provider.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Brave Search API key
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
        }
    }

    /// Create a provider with a custom HTTP client.
    #[must_use]
    pub fn with_client(client: reqwest::Client, api_key: impl Into<String>) -> Self {
        Self {
            client,
            api_key: api_key.into(),
        }
    }
}

/// Brave Search API response structures
mod brave_api {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct BraveSearchResponse {
        pub query: Option<BraveQuery>,
        pub web: Option<BraveWebResults>,
    }

    #[derive(Debug, Deserialize)]
    pub struct BraveQuery {
        pub original: String,
    }

    #[derive(Debug, Deserialize)]
    pub struct BraveWebResults {
        pub results: Vec<BraveWebResult>,
    }

    #[derive(Debug, Deserialize)]
    pub struct BraveWebResult {
        pub title: String,
        pub url: String,
        pub description: Option<String>,
        pub age: Option<String>,
    }
}

#[async_trait]
impl SearchProvider for BraveSearchProvider {
    async fn search(&self, query: &str, max_results: usize) -> Result<SearchResponse> {
        let url = "https://api.search.brave.com/res/v1/web/search";

        let response = self
            .client
            .get(url)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .query(&[
                ("q", query),
                ("count", &max_results.to_string()),
                ("text_decorations", "false"),
            ])
            .send()
            .await
            .context("Failed to send request to Brave Search API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Brave Search API error: {status} - {body}");
        }

        let brave_response: brave_api::BraveSearchResponse = response
            .json()
            .await
            .context("Failed to parse Brave Search API response")?;

        let results = brave_response
            .web
            .map(|web| {
                web.results
                    .into_iter()
                    .map(|r| SearchResult {
                        title: r.title,
                        url: r.url,
                        snippet: r.description.unwrap_or_default(),
                        published_date: r.age,
                    })
                    .collect()
            })
            .unwrap_or_default();

        let query_str = brave_response
            .query
            .map_or_else(|| query.to_string(), |q| q.original);

        Ok(SearchResponse {
            query: query_str,
            results,
            total_results: None,
        })
    }

    fn provider_name(&self) -> &'static str {
        "brave"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult {
            title: "Test Title".into(),
            url: "https://example.com".into(),
            snippet: "Test snippet".into(),
            published_date: Some("2024-01-01".into()),
        };

        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("Test Title"));
        assert!(json.contains("example.com"));
    }

    #[test]
    fn test_search_response_serialization() {
        let response = SearchResponse {
            query: "test query".into(),
            results: vec![SearchResult {
                title: "Result 1".into(),
                url: "https://example.com/1".into(),
                snippet: "First result".into(),
                published_date: None,
            }],
            total_results: Some(100),
        };

        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains("test query"));
        assert!(json.contains("Result 1"));
    }

    #[test]
    fn test_brave_provider_creation() {
        let provider = BraveSearchProvider::new("test-api-key");
        assert_eq!(provider.provider_name(), "brave");
    }

    #[test]
    fn test_brave_provider_with_custom_client() {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("build client");

        let provider = BraveSearchProvider::with_client(client, "test-api-key");
        assert_eq!(provider.provider_name(), "brave");
    }
}
