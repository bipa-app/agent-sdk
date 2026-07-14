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

/// Select the rates that apply to a call of `input_tokens`: the highest tier
/// the call reaches, or the base rates when it reaches none.
///
/// The bound is inclusive — see [`PricingTier::min_input_tokens`]. `tiers` need
/// not be sorted.
#[must_use]
pub fn applicable_pricing(base: Pricing, tiers: &[PricingTier], input_tokens: u32) -> Pricing {
    tiers
        .iter()
        .filter(|tier| input_tokens >= tier.min_input_tokens)
        .max_by_key(|tier| tier.min_input_tokens)
        .map_or(base, |tier| tier.pricing)
}

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
    /// Base pricing distilled from the feed, when present. Applies to a call
    /// whose context stays inside the base band — see [`pricing_tiers`](Self::pricing_tiers).
    pub pricing: Option<Pricing>,
    /// Context-tiered pricing, ascending by threshold, when the feed publishes
    /// it. Empty for a model priced at one flat rate.
    pub pricing_tiers: Vec<PricingTier>,
    /// Whether the model is a reasoning/thinking model, when the feed reports it.
    pub supports_thinking: Option<bool>,
}

/// A long-context price band: from `min_input_tokens` upwards, the provider
/// bills the whole call at `pricing` instead of the base rates.
///
/// Frontier models price long context at a premium — GPT-5.x doubles its input
/// rate above 272K tokens, Gemini and Claude above 200K — so a source that
/// publishes tiers must be billed through them, or a long-context run is priced
/// at a fraction of what it costs and sails past its budget.
///
/// The bound is **inclusive** on both feeds, confirmed against each one's
/// documentation rather than inferred from a field name. `OpenRouter`'s
/// provider docs state a tier "applies when input tokens meet or exceed the
/// `min_context` value", so `min_prompt_tokens` maps straight across. The
/// models.dev SDK schema documents `tier.size` as "the context size at which
/// this tier starts to apply" / "pricing that applies from a given context size
/// upward" — also inclusive, so its `size` maps straight across too. (The
/// models.dev `context_over_<n>` field reads exclusive, but the schema marks it
/// a legacy projection to be superseded by `tiers`, so it does not define the
/// bound.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PricingTier {
    /// The tier applies to a call whose input tokens reach this count.
    pub min_input_tokens: u32,
    /// The rates that replace the base rates for such a call. A source that
    /// restates only some rates for the band leaves the rest inheriting the
    /// base row, so this is the band's *effective* pricing, already merged.
    pub pricing: Pricing,
}

/// Merge a band's stated rates over the base row: a rate the band restates
/// wins, one it omits keeps its base value.
///
/// Both feeds publish partial bands — `OpenRouter`'s Gemini 2.5 Pro override
/// raises `prompt`, `completion` and `input_cache_read` above 200K but says
/// nothing about `input_cache_write`, and models.dev's tier for the same model
/// omits `cache_write` too. Two independent feeds mirroring the vendor's table
/// with the same omission is the vendor not restating that rate for the band,
/// not the rate vanishing: the base value still applies. Inheriting it is
/// therefore what the data means, and it is also the only reading that cannot
/// silently under-bill a component.
#[must_use]
pub fn merge_band_over_base(base: Option<Pricing>, band: Pricing) -> Pricing {
    let Some(base) = base else {
        return band;
    };
    Pricing {
        input: band.input.or(base.input),
        output: band.output.or(base.output),
        cached_input: band.cached_input.or(base.cached_input),
        cache_write: band.cache_write.or(base.cache_write),
        reasoning: band.reasoning.or(base.reasoning),
        notes: band.notes.or(base.notes),
    }
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
