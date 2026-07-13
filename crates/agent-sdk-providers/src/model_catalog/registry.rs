use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::{CatalogEntry, ModelCatalogSource, PricingTier, applicable_pricing};
use crate::model_capabilities::{Pricing, get_model_capabilities};
use agent_sdk_foundation::llm::Usage;

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
    /// Base pricing for cost estimation, if known. Applies to a call whose
    /// context stays inside the base band — see [`pricing_tiers`](Self::pricing_tiers).
    pub pricing: Option<Pricing>,
    /// Long-context price bands, when the source publishes them. Empty for a
    /// model priced at one flat rate (every static-table row).
    pub pricing_tiers: Vec<PricingTier>,
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
        if let Some(resolved) = self.resolve_dynamic(provider, model) {
            return resolved;
        }

        if let Some(caps) = get_model_capabilities(provider, model) {
            return ResolvedModel {
                context_window: caps.context_window,
                max_output_tokens: caps.max_output_tokens,
                pricing: caps.pricing,
                // The compiled-in table prices every model at one flat rate.
                pricing_tiers: Vec::new(),
                supports_thinking: Some(caps.supports_thinking),
                source: ResolvedSource::Static,
            };
        }

        ResolvedModel {
            context_window: None,
            max_output_tokens: None,
            pricing: None,
            pricing_tiers: Vec::new(),
            supports_thinking: None,
            source: ResolvedSource::Unknown,
        }
    }

    /// Resolve through the *dynamic* layers only — user override → feed cache
    /// — reporting `None` when neither carries the pair.
    ///
    /// Unlike [`resolve`](Self::resolve) this never falls back to the static
    /// table, which lets a caller that owns its own static lookup keep the two
    /// sources apart. That distinction matters when the key a model is filed
    /// under differs per source: a caller can then ask this catalog about
    /// *every* key the feeds might use before letting the static table answer
    /// under the (narrower) set of keys it is built from, instead of having
    /// the first key that happens to hit the static layer short-circuit the
    /// search.
    #[must_use]
    pub fn resolve_dynamic(&self, provider: &str, model: &str) -> Option<ResolvedModel> {
        let key = model_key(provider, model);

        if let Some(entry) = self.overrides.get(&key) {
            return Some(resolved_from_entry(entry, ResolvedSource::Override));
        }

        let cache = self.feed_cache.read().ok()?;
        cache
            .get(&key)
            .map(|entry| resolved_from_entry(entry, ResolvedSource::Feed))
    }

    /// Estimate request cost in USD using the layered pricing (override → feed
    /// → static), if any.
    ///
    /// Reports `None` — "this catalog cannot price the call" — rather than a
    /// partial figure when the resolved pricing is missing a rate for a usage
    /// component with a nonzero token count (e.g. a feed row that lists an
    /// input rate but no output rate). A caller can then fall back to another
    /// pricing source instead of billing those tokens at zero.
    ///
    /// A call whose context exceeds a published tier threshold is billed at
    /// that tier's rates, not the base rates.
    #[must_use]
    pub fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64> {
        estimate_resolved(&self.resolve(provider, model), usage)
    }

    /// Estimate request cost in USD from the *dynamic* layers only (override →
    /// feed), never the static table. See [`resolve_dynamic`](Self::resolve_dynamic).
    ///
    /// Declines partial pricing, and selects the tier the call falls in,
    /// exactly as [`estimate_cost_usd`](Self::estimate_cost_usd) does.
    #[must_use]
    pub fn estimate_dynamic_cost_usd(
        &self,
        provider: &str,
        model: &str,
        usage: &Usage,
    ) -> Option<f64> {
        estimate_resolved(&self.resolve_dynamic(provider, model)?, usage)
    }
}

/// Price `usage` from a resolved model: pick the band the call's context falls
/// in, then decline if that band cannot price every component the call bills.
fn estimate_resolved(resolved: &ResolvedModel, usage: &Usage) -> Option<f64> {
    let pricing = applicable_pricing(
        resolved.pricing?,
        &resolved.pricing_tiers,
        usage.input_tokens,
    );
    if !prices_every_billed_component(&pricing, usage) {
        return None;
    }
    pricing.estimate_cost_usd(usage)
}

/// Whether `pricing` carries a rate for every usage component with a nonzero
/// token count.
///
/// Feed rows are not guaranteed to be complete: a row that lists an input
/// rate but no output rate would otherwise price a call's output tokens —
/// typically the majority of its cost — at zero. Under-reporting is worse
/// than declining: a caller that falls back to another pricing source (e.g.
/// the static capability table) recovers the true cost, whereas a partial
/// estimate silently understates spend and delays or defeats a cost budget.
///
/// Cache-read and cache-write tokens count as priced by their own rate *or* by
/// the plain input rate, matching how [`Pricing::estimate_cost_usd`] bills
/// them: both are input tokens, so the input rate is the approximation used
/// when a source publishes no cache rate of its own. What no source may do is
/// bill a component at zero.
fn prices_every_billed_component(pricing: &Pricing, usage: &Usage) -> bool {
    let cache_read_tokens = usage.cached_input_tokens.min(usage.input_tokens);
    let after_cache_read = usage.input_tokens.saturating_sub(cache_read_tokens);
    let cache_write_tokens = usage.cache_creation_input_tokens.min(after_cache_read);
    let plain_input_tokens = after_cache_read.saturating_sub(cache_write_tokens);

    let input_priced = plain_input_tokens == 0 || pricing.input.is_some();
    let cache_read_priced =
        cache_read_tokens == 0 || pricing.cached_input.is_some() || pricing.input.is_some();
    let cache_write_priced =
        cache_write_tokens == 0 || pricing.cache_write.is_some() || pricing.input.is_some();
    let output_priced = usage.output_tokens == 0 || pricing.output.is_some();

    input_priced && cache_read_priced && cache_write_priced && output_priced
}

fn resolved_from_entry(entry: &CatalogEntry, source: ResolvedSource) -> ResolvedModel {
    ResolvedModel {
        context_window: entry.context_window,
        max_output_tokens: entry.max_output_tokens,
        pricing: entry.pricing,
        pricing_tiers: entry.pricing_tiers.clone(),
        supports_thinking: entry.supports_thinking,
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::modelsdev::parse_modelsdev;
    use async_trait::async_trait;

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
                pricing_tiers: Vec::new(),
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

    #[tokio::test]
    async fn dynamic_lookups_never_answer_from_the_static_table() -> Result<()> {
        let source = StaticSource(parse_modelsdev(MODELSDEV_FIXTURE)?);
        let registry = ModelRegistry::new();
        registry.refresh(&source).await?;

        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };

        // `gpt-4o` is in the static table and in neither dynamic layer.
        assert!(registry.resolve("openai", "gpt-4o").pricing.is_some());
        assert!(registry.resolve_dynamic("openai", "gpt-4o").is_none());
        assert!(
            registry
                .estimate_dynamic_cost_usd("openai", "gpt-4o", &usage)
                .is_none(),
            "the dynamic estimate must not fall back to the static table",
        );

        // A model the feed carries is priced by both lookups.
        assert_eq!(
            registry
                .resolve_dynamic("openai", "gpt-5.2")
                .context("the feed carries gpt-5.2")?
                .source,
            ResolvedSource::Feed
        );
        assert!(
            registry
                .estimate_dynamic_cost_usd("openai", "gpt-5.2", &usage)
                .is_some()
        );
        Ok(())
    }

    /// Cache-creation tokens are a component of `input_tokens`, and providers
    /// charge a premium to write the cache (Anthropic: 1.25× input). Billing
    /// them at the ordinary input rate under-reports every cache-warming call.
    #[test]
    fn cache_creation_tokens_bill_at_the_cache_write_rate() -> Result<()> {
        let registry = ModelRegistry::new().with_override(
            "anthropic",
            "claude-haiku-4-5",
            CatalogEntry {
                provider: "anthropic".to_owned(),
                model_id: "claude-haiku-4-5".to_owned(),
                context_window: None,
                max_output_tokens: None,
                // models.dev: input 1, output 5, cache_read 0.1, cache_write 1.25.
                pricing: Some(Pricing::flat_with_cached(1.0, 5.0, 0.1).with_cache_write(1.25)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            },
        );

        // 1M input tokens = 200K plain + 300K cache-read + 500K cache-write,
        // plus 1M output.
        // 0.2*1 + 0.3*0.1 + 0.5*1.25 + 1*5 = 0.2 + 0.03 + 0.625 + 5 = 5.855.
        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cached_input_tokens: 300_000,
            cache_creation_input_tokens: 500_000,
        };
        let cost = registry
            .estimate_cost_usd("anthropic", "claude-haiku-4-5", &usage)
            .context("cost estimate missing")?;
        assert!((cost - 5.855).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    /// A source with no cache-write rate bills those tokens at the ordinary
    /// input rate — the provider-agnostic approximation the compiled-in table
    /// has always used — rather than declining or billing them at zero.
    #[test]
    fn cache_creation_falls_back_to_the_input_rate() -> Result<()> {
        let registry = ModelRegistry::new().with_override(
            "anthropic",
            "no-write-rate",
            CatalogEntry {
                provider: "anthropic".to_owned(),
                model_id: "no-write-rate".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(Pricing::flat(1.0, 5.0)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            },
        );
        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 500_000,
        };
        // Every input token at $1/M, cache-write included: $1.00.
        let cost = registry
            .estimate_cost_usd("anthropic", "no-write-rate", &usage)
            .context("cost estimate missing")?;
        assert!((cost - 1.0).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    /// Long-context calls are billed at the tier the call falls in. Frontier
    /// models roughly double their rates above the threshold, so pricing an
    /// over-threshold call at the base rates halves the estimate.
    #[test]
    fn long_context_calls_bill_at_the_tier_rate() -> Result<()> {
        let registry = ModelRegistry::new().with_override(
            "openai",
            "gpt-5.4",
            CatalogEntry {
                provider: "openai".to_owned(),
                model_id: "gpt-5.4".to_owned(),
                context_window: None,
                max_output_tokens: None,
                // models.dev: base 2.5/15, tier above 272K context 5/22.5.
                pricing: Some(Pricing::flat(2.5, 15.0)),
                pricing_tiers: vec![PricingTier {
                    min_context_tokens: 272_000,
                    pricing: Pricing::flat(5.0, 22.5),
                }],
                supports_thinking: None,
            },
        );

        // Inside the base band: 200K in + 100K out = 0.2*2.5 + 0.1*15 = 2.0.
        let short = Usage {
            input_tokens: 200_000,
            output_tokens: 100_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        let short_cost = registry
            .estimate_cost_usd("openai", "gpt-5.4", &short)
            .context("cost estimate missing")?;
        assert!(
            (short_cost - 2.0).abs() < 1e-9,
            "unexpected cost: {short_cost}"
        );

        // Past the threshold: 300K in + 100K out = 0.3*5 + 0.1*22.5 = 3.75,
        // where the base rates would have said 0.3*2.5 + 0.1*15 = 2.25.
        let long = Usage {
            input_tokens: 300_000,
            output_tokens: 100_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        let long_cost = registry
            .estimate_cost_usd("openai", "gpt-5.4", &long)
            .context("cost estimate missing")?;
        assert!(
            (long_cost - 3.75).abs() < 1e-9,
            "unexpected cost: {long_cost}"
        );
        Ok(())
    }

    #[test]
    fn estimate_cost_usd_declines_a_row_missing_a_billed_rate() -> Result<()> {
        // A feed row that lists an input rate but no output rate: pricing the
        // call from it would bill the output tokens at zero.
        let registry = ModelRegistry::new().with_override(
            "openai",
            "gpt-4o",
            CatalogEntry {
                provider: "openai".to_owned(),
                model_id: "gpt-4o".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(Pricing {
                    input: Some(crate::model_capabilities::PricePoint::new(1.0)),
                    output: None,
                    cached_input: None,
                    cache_write: None,
                    notes: None,
                }),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            },
        );

        let with_output = Usage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        assert!(
            registry
                .estimate_cost_usd("openai", "gpt-4o", &with_output)
                .is_none(),
            "a row with no output rate must decline a call that bills output tokens",
        );

        // The same row still prices a call that bills no output tokens.
        let input_only = Usage {
            input_tokens: 2_000,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        let cost = registry
            .estimate_cost_usd("openai", "gpt-4o", &input_only)
            .context("input-only usage is fully priced by an input rate")?;
        assert!((cost - 0.002).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }
}
