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

    pub(crate) struct StaticSource(pub(crate) Vec<CatalogEntry>);

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
