//! Web tools for search and fetching.
//!
//! This module provides tools for web operations:
//!
//! - [`WebSearchTool`] - Search the web using pluggable providers
//! - [`SearchProvider`] - Trait for search provider implementations
//! - [`BraveSearchProvider`] - Brave Search API integration
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::web::{WebSearchTool, BraveSearchProvider};
//!
//! let provider = BraveSearchProvider::new(api_key);
//! let search_tool = WebSearchTool::new(provider);
//!
//! tools.register(search_tool);
//! ```

pub mod provider;
pub mod search;

pub use provider::{BraveSearchProvider, SearchProvider, SearchResponse, SearchResult};
pub use search::WebSearchTool;
