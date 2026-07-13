use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::{CatalogEntry, ModelCatalogSource};
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
        if let Some(resolved) = self.resolve_dynamic(provider, model) {
            return resolved;
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
    #[must_use]
    pub fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64> {
        estimate_from_pricing(self.resolve(provider, model).pricing?, usage)
    }

    /// Estimate request cost in USD from the *dynamic* layers only (override →
    /// feed), never the static table. See [`resolve_dynamic`](Self::resolve_dynamic).
    ///
    /// Declines partial pricing exactly as [`estimate_cost_usd`](Self::estimate_cost_usd) does.
    #[must_use]
    pub fn estimate_dynamic_cost_usd(
        &self,
        provider: &str,
        model: &str,
        usage: &Usage,
    ) -> Option<f64> {
        estimate_from_pricing(self.resolve_dynamic(provider, model)?.pricing?, usage)
    }
}

/// Price `usage` from `pricing`, declining when a billed component has no rate.
fn estimate_from_pricing(pricing: Pricing, usage: &Usage) -> Option<f64> {
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
/// Cached input tokens are considered priced by either a cache rate or the
/// plain input rate, matching how [`Pricing::estimate_cost_usd`] bills them.
fn prices_every_billed_component(pricing: &Pricing, usage: &Usage) -> bool {
    let cached_input_tokens = usage.cached_input_tokens.min(usage.input_tokens);
    let uncached_input_tokens = usage.input_tokens.saturating_sub(cached_input_tokens);

    let input_priced = uncached_input_tokens == 0 || pricing.input.is_some();
    let cached_priced =
        cached_input_tokens == 0 || pricing.cached_input.is_some() || pricing.input.is_some();
    let output_priced = usage.output_tokens == 0 || pricing.output.is_some();

    input_priced && cached_priced && output_priced
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
                    notes: None,
                }),
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
