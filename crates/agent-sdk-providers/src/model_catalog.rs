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

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use agent_sdk_foundation::llm::Usage;
use anyhow::{Context, Result};
use async_trait::async_trait;

use crate::model_capabilities::{PricePoint, Pricing, get_model_capabilities};

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/models";
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

/// Map a models.dev provider key to our internal provider name.
///
/// models.dev keys Google models under `google`; we call that provider
/// `gemini`. `anthropic` and `openai` already match.
fn map_modelsdev_provider(key: &str) -> String {
    match key {
        "google" => "gemini".to_owned(),
        other => other.to_owned(),
    }
}

/// Build a [`Pricing`] from a models.dev `cost` object (all USD per million).
fn pricing_from_modelsdev_cost(cost: &ModelsDevCost) -> Option<Pricing> {
    if cost.input.is_none() && cost.output.is_none() && cost.cache_read.is_none() {
        return None;
    }
    Some(Pricing {
        input: cost.input.map(PricePoint::new),
        output: cost.output.map(PricePoint::new),
        cached_input: cost.cache_read.map(PricePoint::new),
        // `notes` is `&'static str`; feed-derived rows carry no static note.
        notes: None,
    })
}

#[derive(serde::Deserialize)]
struct ModelsDevCost {
    #[serde(default)]
    input: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
    #[serde(default)]
    cache_read: Option<f64>,
    // `cache_write` exists in the feed but we do not model a separate
    // cache-write price, so it is intentionally not deserialized.
}

#[derive(serde::Deserialize)]
struct ModelsDevLimit {
    #[serde(default)]
    context: Option<u32>,
    #[serde(default)]
    output: Option<u32>,
}

#[derive(serde::Deserialize)]
struct ModelsDevModel {
    id: String,
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    cost: Option<ModelsDevCost>,
    #[serde(default)]
    limit: Option<ModelsDevLimit>,
}

#[derive(serde::Deserialize)]
struct ModelsDevProvider {
    #[serde(default)]
    models: HashMap<String, ModelsDevModel>,
}

/// Parse the models.dev `api.json` body into catalog entries.
///
/// The document is a JSON object keyed by provider id; each provider has a
/// `models` object keyed by bare model id. Each model carries `cost` (USD per
/// million), `limit` (`context` / `output`), and `reasoning` (bool).
///
/// # Errors
///
/// Returns an error if the body is not valid models.dev JSON.
pub fn parse_modelsdev(json: &str) -> Result<Vec<CatalogEntry>> {
    let providers: HashMap<String, ModelsDevProvider> =
        serde_json::from_str(json).context("failed to parse models.dev api.json")?;
    let mut entries = Vec::new();
    for (provider_key, provider_obj) in providers {
        let provider = map_modelsdev_provider(&provider_key);
        for model in provider_obj.models.into_values() {
            let pricing = model.cost.as_ref().and_then(pricing_from_modelsdev_cost);
            let (context_window, max_output_tokens) = model
                .limit
                .map_or((None, None), |limit| (limit.context, limit.output));
            entries.push(CatalogEntry {
                provider: provider.clone(),
                model_id: model.id,
                context_window,
                max_output_tokens,
                pricing,
                supports_thinking: model.reasoning,
            });
        }
    }
    Ok(entries)
}

/// The default capability/pricing feed: <https://models.dev/api.json>.
pub struct ModelsDevSource {
    client: reqwest::Client,
    url: String,
}

impl Default for ModelsDevSource {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelsDevSource {
    /// Create a source pointing at the canonical models.dev endpoint.
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: build_feed_client(),
            url: MODELS_DEV_URL.to_owned(),
        }
    }

    /// Override the feed URL (e.g. for a mirror or a local test server).
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }
}

#[async_trait]
impl ModelCatalogSource for ModelsDevSource {
    async fn fetch(&self) -> Result<Vec<CatalogEntry>> {
        let body = self
            .client
            .get(&self.url)
            .send()
            .await
            .context("models.dev request failed")?
            .error_for_status()
            .context("models.dev returned an error status")?
            .text()
            .await
            .context("failed to read models.dev body")?;
        parse_modelsdev(&body)
    }
}

#[derive(serde::Deserialize)]
struct OpenRouterPricing {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
}

#[derive(serde::Deserialize)]
struct OpenRouterTopProvider {
    #[serde(default)]
    max_completion_tokens: Option<u32>,
}

#[derive(serde::Deserialize)]
struct OpenRouterModel {
    id: String,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    pricing: Option<OpenRouterPricing>,
    #[serde(default)]
    top_provider: Option<OpenRouterTopProvider>,
}

#[derive(serde::Deserialize)]
struct OpenRouterResponse {
    #[serde(default)]
    data: Vec<OpenRouterModel>,
}

/// Convert an `OpenRouter` USD-per-token price string to USD per million tokens.
///
/// `OpenRouter` uses `"-1"` (and other non-positive values) as a sentinel for
/// "no fixed price" on routed/aggregate models like `openrouter/auto`. Those
/// must be treated as *absent* pricing, not real negative prices — a negative
/// `PricePoint` would make [`estimate_cost_usd`](ModelRegistry::estimate_cost_usd)
/// return a negative cost. Any value that is unparseable, non-finite, or `<= 0`
/// yields `None`.
fn openrouter_price_per_million(value: &str) -> Option<PricePoint> {
    let per_token: f64 = value.trim().parse().ok()?;
    if !per_token.is_finite() || per_token <= 0.0 {
        return None;
    }
    Some(PricePoint::new(per_token * 1_000_000.0))
}

/// Map an `OpenRouter` namespaced id (`vendor/model`) to our `(provider, model)`.
///
/// `OpenRouter` vendors line up with our provider names except for `google`,
/// which we call `gemini`. Ids without a slash fall back to the whole id as the
/// model with an empty provider.
///
/// # Keying limitation
///
/// The resulting `model` segment is `OpenRouter`'s own slug, which does **not**
/// always match the model id our chat endpoints expect:
///
/// - `OpenRouter` slugs use dots where the first-party providers use dashes
///   (e.g. `claude-opus-4.8` here vs `claude-opus-4-8` on the Anthropic API),
///   and they sometimes carry extra route suffixes (`:free`, `:beta`,
///   `-thinking`).
/// - Aggregate/routed ids such as `openrouter/auto` have no first-party
///   equivalent at all.
///
/// We deliberately do **not** rewrite the slug here: there is no lossless,
/// general normalization back to each provider's native id, so a guessed
/// mapping would silently mis-key entries. As a result, a model whose slug
/// differs from its native id will not be resolved by [`ModelRegistry`] when
/// looked up by the native `(provider, model)` pair. Prefer
/// [`ModelsDevSource`], which keys on bare native ids, when feed coverage of
/// first-party models matters; use [`OpenRouterSource`] mainly for models
/// routed through `OpenRouter` under their `OpenRouter` slug.
fn split_openrouter_id(id: &str) -> (String, String) {
    match id.split_once('/') {
        Some((vendor, model)) => {
            let provider = match vendor {
                "google" => "gemini",
                other => other,
            };
            (provider.to_owned(), model.to_owned())
        }
        None => (String::new(), id.to_owned()),
    }
}

/// Parse the `OpenRouter` `/models` body into catalog entries.
///
/// The document is `{ "data": [ ... ] }`; each model has a namespaced `id`
/// (`vendor/model`), a `context_length`, and a `pricing` object whose values
/// are USD-per-token strings (multiplied by 1e6 to recover USD per million).
///
/// # Errors
///
/// Returns an error if the body is not valid `OpenRouter` JSON.
pub fn parse_openrouter(json: &str) -> Result<Vec<CatalogEntry>> {
    let parsed: OpenRouterResponse =
        serde_json::from_str(json).context("failed to parse OpenRouter models response")?;
    Ok(parsed
        .data
        .into_iter()
        .map(|model| {
            let (provider, model_id) = split_openrouter_id(&model.id);
            let pricing = model.pricing.and_then(|p| {
                let input = p.prompt.as_deref().and_then(openrouter_price_per_million);
                let output = p
                    .completion
                    .as_deref()
                    .and_then(openrouter_price_per_million);
                let cached_input = p
                    .input_cache_read
                    .as_deref()
                    .and_then(openrouter_price_per_million);
                if input.is_none() && output.is_none() && cached_input.is_none() {
                    None
                } else {
                    Some(Pricing {
                        input,
                        output,
                        cached_input,
                        notes: None,
                    })
                }
            });
            let max_output_tokens = model.top_provider.and_then(|tp| tp.max_completion_tokens);
            CatalogEntry {
                provider,
                model_id,
                context_window: model.context_length,
                max_output_tokens,
                pricing,
                supports_thinking: None,
            }
        })
        .collect())
}

/// An alternative public feed: <https://openrouter.ai/api/v1/models> (no key).
///
/// Note: `OpenRouter` keys models by its own `vendor/model` slug, which does not
/// always match the native id our chat endpoints expect (dotted vs dashed
/// versions, route suffixes, aggregate ids like `openrouter/auto`). The
/// internal slug-splitting helper documents the keying limitation and why we do
/// not rewrite slugs. Prefer [`ModelsDevSource`] for first-party model coverage.
pub struct OpenRouterSource {
    client: reqwest::Client,
    url: String,
}

impl Default for OpenRouterSource {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterSource {
    /// Create a source pointing at the canonical `OpenRouter` models endpoint.
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: build_feed_client(),
            url: OPENROUTER_URL.to_owned(),
        }
    }

    /// Override the feed URL (e.g. for a mirror or a local test server).
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }
}

#[async_trait]
impl ModelCatalogSource for OpenRouterSource {
    async fn fetch(&self) -> Result<Vec<CatalogEntry>> {
        let body = self
            .client
            .get(&self.url)
            .send()
            .await
            .context("OpenRouter request failed")?
            .error_for_status()
            .context("OpenRouter returned an error status")?
            .text()
            .await
            .context("failed to read OpenRouter body")?;
        parse_openrouter(&body)
    }
}

fn build_feed_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(FEED_TIMEOUT_SECS))
        .timeout(Duration::from_secs(FEED_TIMEOUT_SECS))
        .build()
        .unwrap_or_default()
}

/// Where a [`ResolvedModel`] came from, in the registry's resolution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedSource {
    /// A user-registered override.
    Override,
    /// The refreshable third-party feed cache.
    Feed,
    /// The static [`model_capabilities`](crate::model_capabilities) table.
    Static,
    /// No source had this model; fields are empty defaults.
    Unknown,
}

/// The resolved capability/pricing view of a provider/model.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModel {
    /// Maximum total context window in tokens, if known.
    pub context_window: Option<u32>,
    /// Maximum output tokens per response, if known.
    pub max_output_tokens: Option<u32>,
    /// Pricing for cost estimation, if known.
    pub pricing: Option<Pricing>,
    /// Whether the model is a reasoning/thinking model, if known.
    pub supports_thinking: Option<bool>,
    /// Which layer satisfied the lookup.
    pub source: ResolvedSource,
}

type ModelKey = (String, String);

fn model_key(provider: &str, model: &str) -> ModelKey {
    (provider.to_ascii_lowercase(), model.to_ascii_lowercase())
}

/// A layered model resolver: user override → feed cache → static table.
///
/// Construct one with [`new`](Self::new), optionally seed user overrides, then
/// [`refresh`](Self::refresh) it from a [`ModelCatalogSource`] to populate the
/// feed cache. [`resolve`](Self::resolve) walks the layers in priority order
/// and always returns a [`ResolvedModel`] (empty + [`ResolvedSource::Unknown`]
/// when nothing matched).
#[derive(Clone, Default)]
pub struct ModelRegistry {
    overrides: HashMap<ModelKey, CatalogEntry>,
    feed_cache: Arc<RwLock<HashMap<ModelKey, CatalogEntry>>>,
}

impl ModelRegistry {
    /// Create an empty registry (no overrides, no feed loaded).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style: add a user override that wins over feed and static data.
    #[must_use]
    pub fn with_override(
        mut self,
        provider: impl AsRef<str>,
        model: impl AsRef<str>,
        entry: CatalogEntry,
    ) -> Self {
        self.overrides
            .insert(model_key(provider.as_ref(), model.as_ref()), entry);
        self
    }

    /// Register (or replace) a user override in place.
    pub fn register(
        &mut self,
        provider: impl AsRef<str>,
        model: impl AsRef<str>,
        entry: CatalogEntry,
    ) {
        self.overrides
            .insert(model_key(provider.as_ref(), model.as_ref()), entry);
    }

    /// Fetch from `source` and replace the feed cache. Returns the entry count.
    ///
    /// # Errors
    ///
    /// Returns an error if the source fails to fetch or parse, or if the cache
    /// lock is poisoned.
    pub async fn refresh(&self, source: &dyn ModelCatalogSource) -> Result<usize> {
        let entries = source.fetch().await?;
        let mut cache = self
            .feed_cache
            .write()
            .ok()
            .context("feed cache lock poisoned")?;
        cache.clear();
        for entry in entries {
            cache.insert(model_key(&entry.provider, &entry.model_id), entry);
        }
        Ok(cache.len())
    }

    /// Resolve a provider/model through the layers: override → feed → static.
    #[must_use]
    pub fn resolve(&self, provider: &str, model: &str) -> ResolvedModel {
        let key = model_key(provider, model);

        if let Some(entry) = self.overrides.get(&key) {
            return resolved_from_entry(entry, ResolvedSource::Override);
        }

        if let Ok(cache) = self.feed_cache.read()
            && let Some(entry) = cache.get(&key)
        {
            return resolved_from_entry(entry, ResolvedSource::Feed);
        }

        if let Some(caps) = get_model_capabilities(provider, model) {
            return ResolvedModel {
                context_window: caps.context_window,
                max_output_tokens: caps.max_output_tokens,
                pricing: caps.pricing,
                supports_thinking: Some(caps.supports_thinking),
                source: ResolvedSource::Static,
            };
        }

        ResolvedModel {
            context_window: None,
            max_output_tokens: None,
            pricing: None,
            supports_thinking: None,
            source: ResolvedSource::Unknown,
        }
    }

    /// Estimate request cost in USD using the resolved pricing, if any.
    #[must_use]
    pub fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64> {
        self.resolve(provider, model)
            .pricing
            .and_then(|pricing| pricing.estimate_cost_usd(usage))
    }
}

const fn resolved_from_entry(entry: &CatalogEntry, source: ResolvedSource) -> ResolvedModel {
    ResolvedModel {
        context_window: entry.context_window,
        max_output_tokens: entry.max_output_tokens,
        pricing: entry.pricing,
        supports_thinking: entry.supports_thinking,
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trimmed real-shape models.dev fixture: one model each for the three
    // providers we map, including the `google` -> `gemini` rename.
    const MODELSDEV_FIXTURE: &str = r#"{
      "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
          "claude-sonnet-4-5": {
            "id": "claude-sonnet-4-5",
            "name": "Claude Sonnet 4.5",
            "reasoning": true,
            "limit": { "context": 1000000, "output": 64000 },
            "cost": { "input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75 }
          }
        }
      },
      "openai": {
        "id": "openai",
        "name": "OpenAI",
        "models": {
          "gpt-5.2": {
            "id": "gpt-5.2",
            "name": "GPT-5.2",
            "reasoning": true,
            "limit": { "context": 400000, "output": 128000 },
            "cost": { "input": 1.75, "output": 14, "cache_read": 0.175 }
          }
        }
      },
      "google": {
        "id": "google",
        "name": "Google",
        "models": {
          "gemini-2.5-pro": {
            "id": "gemini-2.5-pro",
            "name": "Gemini 2.5 Pro",
            "reasoning": true,
            "limit": { "context": 1048576, "output": 65536 },
            "cost": { "input": 1.25, "output": 10, "cache_read": 0.31, "cache_write": 2.375 }
          }
        }
      }
    }"#;

    const OPENROUTER_FIXTURE: &str = r#"{
      "data": [
        {
          "id": "anthropic/claude-opus-4.8",
          "name": "Anthropic: Claude Opus 4.8",
          "context_length": 1000000,
          "pricing": {
            "prompt": "0.000005",
            "completion": "0.000025",
            "input_cache_read": "0.0000005"
          },
          "top_provider": { "max_completion_tokens": 128000 }
        },
        {
          "id": "google/gemini-2.5-pro",
          "name": "Google: Gemini 2.5 Pro",
          "context_length": 1048576,
          "pricing": { "prompt": "0.00000125", "completion": "0.00001" },
          "top_provider": { "max_completion_tokens": 65536 }
        }
      ]
    }"#;

    // `openrouter/auto` reports `"-1"` for every price — a sentinel for
    // "no fixed price", not a real negative cost.
    const OPENROUTER_SENTINEL_FIXTURE: &str = r#"{
      "data": [
        {
          "id": "openrouter/auto",
          "name": "Auto Router",
          "context_length": 2000000,
          "pricing": {
            "prompt": "-1",
            "completion": "-1",
            "input_cache_read": "-1"
          }
        }
      ]
    }"#;

    fn find<'a>(entries: &'a [CatalogEntry], provider: &str, model: &str) -> &'a CatalogEntry {
        entries
            .iter()
            .find(|e| e.provider == provider && e.model_id == model)
            .unwrap_or_else(|| panic!("missing {provider}/{model}"))
    }

    #[test]
    fn parse_modelsdev_maps_pricing_limits_and_provider() -> Result<()> {
        let entries = parse_modelsdev(MODELSDEV_FIXTURE)?;
        assert_eq!(entries.len(), 3);

        let claude = find(&entries, "anthropic", "claude-sonnet-4-5");
        assert_eq!(claude.context_window, Some(1_000_000));
        assert_eq!(claude.max_output_tokens, Some(64_000));
        assert_eq!(claude.supports_thinking, Some(true));
        let pricing = claude.pricing.context("claude pricing missing")?;
        assert!(
            (pricing.input.context("input")?.usd_per_million_tokens - 3.0).abs() < f64::EPSILON
        );
        assert!(
            (pricing.output.context("output")?.usd_per_million_tokens - 15.0).abs() < f64::EPSILON
        );
        assert!(
            (pricing
                .cached_input
                .context("cache")?
                .usd_per_million_tokens
                - 0.3)
                .abs()
                < f64::EPSILON
        );

        // `google` is remapped to our `gemini` provider name.
        let gemini = find(&entries, "gemini", "gemini-2.5-pro");
        assert_eq!(gemini.context_window, Some(1_048_576));
        assert_eq!(gemini.max_output_tokens, Some(65_536));

        // No model is keyed under the raw `google` provider name.
        assert!(!entries.iter().any(|e| e.provider == "google"));
        Ok(())
    }

    #[test]
    fn parse_openrouter_converts_per_token_to_per_million_and_splits_ids() -> Result<()> {
        let entries = parse_openrouter(OPENROUTER_FIXTURE)?;
        assert_eq!(entries.len(), 2);

        let opus = find(&entries, "anthropic", "claude-opus-4.8");
        assert_eq!(opus.context_window, Some(1_000_000));
        assert_eq!(opus.max_output_tokens, Some(128_000));
        let pricing = opus.pricing.context("opus pricing missing")?;
        // 0.000005 USD/token * 1e6 = 5.0 USD/M.
        assert!(
            (pricing.input.context("input")?.usd_per_million_tokens - 5.0).abs() < f64::EPSILON
        );
        assert!(
            (pricing.output.context("output")?.usd_per_million_tokens - 25.0).abs() < f64::EPSILON
        );
        assert!(
            (pricing
                .cached_input
                .context("cache")?
                .usd_per_million_tokens
                - 0.5)
                .abs()
                < f64::EPSILON
        );

        // `google/...` is split + remapped to our `gemini` provider name.
        let gemini = find(&entries, "gemini", "gemini-2.5-pro");
        assert_eq!(gemini.context_window, Some(1_048_576));
        Ok(())
    }

    #[tokio::test]
    async fn parse_openrouter_treats_minus_one_sentinel_prices_as_absent() -> Result<()> {
        let entries = parse_openrouter(OPENROUTER_SENTINEL_FIXTURE)?;
        assert_eq!(entries.len(), 1);

        let auto = find(&entries, "openrouter", "auto");
        // The `"-1"` sentinel must collapse to no pricing at all, never a
        // negative `PricePoint` (which would make estimate_cost_usd go negative).
        assert!(
            auto.pricing.is_none(),
            "sentinel `-1` prices must yield None pricing, got {:?}",
            auto.pricing
        );
        // Non-price metadata is still captured.
        assert_eq!(auto.context_window, Some(2_000_000));

        // And a registry built from this feed never returns a negative cost.
        let registry = ModelRegistry::new();
        registry.refresh(&StaticSource(entries)).await?;
        let usage = Usage {
            input_tokens: 1_000,
            output_tokens: 1_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        assert_eq!(
            registry.estimate_cost_usd("openrouter", "auto", &usage),
            None
        );
        Ok(())
    }

    struct StaticSource(Vec<CatalogEntry>);

    #[async_trait]
    impl ModelCatalogSource for StaticSource {
        async fn fetch(&self) -> Result<Vec<CatalogEntry>> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn registry_layered_resolution_prefers_override_then_feed_then_static() -> Result<()> {
        let source = StaticSource(parse_modelsdev(MODELSDEV_FIXTURE)?);
        let registry = ModelRegistry::new().with_override(
            "anthropic",
            "claude-sonnet-4-5",
            CatalogEntry {
                provider: "anthropic".to_owned(),
                model_id: "claude-sonnet-4-5".to_owned(),
                context_window: Some(123),
                max_output_tokens: Some(7),
                pricing: Some(Pricing::flat(1.0, 2.0)),
                supports_thinking: Some(false),
            },
        );

        let count = registry.refresh(&source).await?;
        assert_eq!(count, 3);

        // Override wins even though the feed also has this model.
        let overridden = registry.resolve("anthropic", "claude-sonnet-4-5");
        assert_eq!(overridden.source, ResolvedSource::Override);
        assert_eq!(overridden.context_window, Some(123));

        // Feed satisfies a model only the feed knows.
        let feed = registry.resolve("openai", "gpt-5.2");
        assert_eq!(feed.source, ResolvedSource::Feed);
        assert_eq!(feed.max_output_tokens, Some(128_000));

        // Static table satisfies a model the feed does not carry. `gpt-4o` is
        // in the bundled static table but not in our trimmed fixture.
        let static_hit = registry.resolve("openai", "gpt-4o");
        assert_eq!(static_hit.source, ResolvedSource::Static);
        assert!(static_hit.pricing.is_some());

        // Nothing knows this model.
        let unknown = registry.resolve("openai", "totally-made-up-model");
        assert_eq!(unknown.source, ResolvedSource::Unknown);
        assert!(unknown.pricing.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn estimate_cost_usd_uses_feed_loaded_pricing() -> Result<()> {
        let source = StaticSource(parse_modelsdev(MODELSDEV_FIXTURE)?);
        let registry = ModelRegistry::new();
        registry.refresh(&source).await?;

        // gpt-5.2 from the feed: $1.75/M input, $14/M output, $0.175/M cache_read.
        // 1000 uncached input + 1000 cached input + 1000 output.
        let usage = Usage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            cached_input_tokens: 1_000,
            cache_creation_input_tokens: 0,
        };
        let cost = registry
            .estimate_cost_usd("openai", "gpt-5.2", &usage)
            .context("cost estimate missing")?;
        // (1000/1e6*1.75) + (1000/1e6*0.175) + (1000/1e6*14)
        // = 0.00175 + 0.000175 + 0.014 = 0.015925
        assert!((cost - 0.015_925).abs() < 1e-9);
        Ok(())
    }
}
