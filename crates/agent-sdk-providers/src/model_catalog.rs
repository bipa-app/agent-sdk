//! Dynamic model capability/pricing discovery from third-party feeds.
//!
//! The static [`model_capabilities`](crate::model_capabilities) table is the
//! offline fallback: it ships in the binary and never makes a network call.
//! This module layers a *refreshable* feed on top so newly shipped models gain
//! pricing and limit metadata without an SDK code change.
//!
//! The pieces:
//!
//! - [`CatalogEntry`] — a single provider/model row distilled from a feed.
//! - [`ModelCatalogSource`] — the trait a feed implements; each source pairs an
//!   async `fetch` with a pure `parse` so the parsing is testable offline.
//! - [`ModelsDevSource`] (default) and [`OpenRouterSource`] — two concrete
//!   feeds.
//! - [`ModelRegistry`] — a layered resolver: user override → feed cache →
//!   static table → graceful empty defaults.
//!
//! This whole module is gated behind the `model-discovery` cargo feature
//! because it performs outbound HTTP to a third-party catalog. The default
//! build is unchanged.

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::time::Duration;

use crate::model_capabilities::Pricing;

pub mod modelsdev;
pub mod openrouter;
pub mod registry;

pub use modelsdev::{ModelsDevSource, parse_modelsdev};
pub use openrouter::{OpenRouterSource, parse_openrouter};
pub use registry::{ModelRegistry, ResolvedModel, ResolvedSource};

pub(crate) const MODELS_DEV_URL: &str = "https://models.dev/api.json";
pub(crate) const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/models";
const FEED_TIMEOUT_SECS: u64 = 30;

/// A single provider/model capability + pricing row distilled from a feed.
///
/// Field shapes mirror [`ModelCapabilities`](crate::model_capabilities::ModelCapabilities)
/// so the registry can resolve feed rows and static rows into the same shape.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogEntry {
    /// Our provider name (e.g. `anthropic`, `openai`, `gemini`).
    pub provider: String,
    /// The model identifier as the provider's chat endpoint expects it.
    pub model_id: String,
    /// Maximum total context window in tokens, when the feed reports it.
    pub context_window: Option<u32>,
    /// Maximum output tokens per response, when the feed reports it.
    pub max_output_tokens: Option<u32>,
    /// Pricing distilled from the feed, when present.
    pub pricing: Option<Pricing>,
    /// Whether the model is a reasoning/thinking model, when the feed reports it.
    pub supports_thinking: Option<bool>,
}

/// A third-party feed of [`CatalogEntry`] rows.
#[async_trait]
pub trait ModelCatalogSource: Send + Sync {
    /// Fetch and parse the feed into catalog entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the body cannot be parsed.
    async fn fetch(&self) -> Result<Vec<CatalogEntry>>;
}

pub(crate) fn build_feed_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(FEED_TIMEOUT_SECS))
        .timeout(Duration::from_secs(FEED_TIMEOUT_SECS))
        .build()
        .context("failed to build model-catalog feed HTTP client")
}
