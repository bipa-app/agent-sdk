//! Web tools for search and fetching.
//!
//! This module provides tools for web operations:
//!
//! - [`WebSearchTool`] - Search the web using pluggable providers
//! - [`SearchProvider`] - Trait for search provider implementations
//! - [`BraveSearchProvider`] - Brave Search API integration
//! - [`LinkFetchTool`] - Securely fetch and read web page content
//! - [`UrlValidator`] - URL validation with SSRF protection
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::web::{WebSearchTool, BraveSearchProvider, LinkFetchTool, UrlValidator};
//!
//! // Create a web search tool
//! let provider = BraveSearchProvider::new(api_key);
//! let search_tool = WebSearchTool::new(provider);
//!
//! // Create a link fetch tool with default security
//! let fetch_tool = LinkFetchTool::new();
//!
//! // Or customize the URL validator
//! let validator = UrlValidator::new()
//!     .with_allowed_domains(vec!["example.com".to_string()]);
//! let fetch_tool = LinkFetchTool::new().with_validator(validator);
//! ```

pub mod fetch;
pub mod provider;
pub mod search;
pub mod security;

pub use fetch::{FetchFormat, LinkFetchTool};
pub use provider::{BraveSearchProvider, SearchProvider, SearchResponse, SearchResult};
pub use search::WebSearchTool;
pub use security::UrlValidator;
