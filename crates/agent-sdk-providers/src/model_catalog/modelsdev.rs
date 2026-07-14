use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;

use super::{
    CatalogEntry, MODELS_DEV_URL, ModelCatalogSource, PricingTier, build_feed_client,
    merge_band_over_base,
};
use crate::model_capabilities::{PricePoint, Pricing};

#[derive(serde::Deserialize)]
struct ModelsDevCost {
    #[serde(default)]
    input: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
    #[serde(default)]
    cache_read: Option<f64>,
    #[serde(default)]
    cache_write: Option<f64>,
    #[serde(default)]
    reasoning: Option<f64>,
    /// Long-context price bands. The feed publishes these for frontier models
    /// (GPT-5.x above 272K input tokens, Gemini / Claude above 200K).
    #[serde(default)]
    tiers: Vec<ModelsDevTier>,
}

#[derive(serde::Deserialize)]
struct ModelsDevTier {
    #[serde(default)]
    input: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
    #[serde(default)]
    cache_read: Option<f64>,
    #[serde(default)]
    cache_write: Option<f64>,
    #[serde(default)]
    tier: Option<ModelsDevTierBound>,
}

/// Every field is optional so that a feed row whose shape has drifted cannot
/// abort the parse of the whole document: one required field missing anywhere
/// in a 3MB body would fail `serde_json::from_str` outright, leaving the
/// catalog empty and every feed-priced model unpriced on the next refresh.
/// A bound that does not arrive intact is simply un-interpretable, and the row
/// carrying it drops its pricing (see [`tiers_from_modelsdev_cost`]).
#[derive(serde::Deserialize)]
struct ModelsDevTierBound {
    /// The only bound the feed publishes today is `context`; a row bounded by
    /// anything else cannot be priced from data this parser understands.
    #[serde(rename = "type", default)]
    bound_type: Option<String>,
    #[serde(default)]
    size: Option<u32>,
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

fn map_modelsdev_provider(key: &str) -> String {
    match key {
        "google" => "gemini".to_owned(),
        other => other.to_owned(),
    }
}

/// A models.dev rate (USD per 1M tokens) is only usable when it is a finite
/// positive number: the feed reports `0` both for genuinely free models and
/// for rows whose price it does not know, and either way a zero rate would
/// price the component as free. Mirrors the `OpenRouter` parser's guard.
fn modelsdev_price_per_million(usd_per_million_tokens: f64) -> Option<PricePoint> {
    (usd_per_million_tokens.is_finite() && usd_per_million_tokens > 0.0)
        .then(|| PricePoint::new(usd_per_million_tokens))
}

fn pricing_from_rates(
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
    reasoning: Option<f64>,
) -> Option<Pricing> {
    let input = input.and_then(modelsdev_price_per_million);
    let output = output.and_then(modelsdev_price_per_million);
    let cached_input = cache_read.and_then(modelsdev_price_per_million);
    let cache_write = cache_write.and_then(modelsdev_price_per_million);
    let reasoning = reasoning.and_then(modelsdev_price_per_million);

    if input.is_none()
        && output.is_none()
        && cached_input.is_none()
        && cache_write.is_none()
        && reasoning.is_none()
    {
        return None;
    }
    Some(Pricing {
        input,
        output,
        cached_input,
        cache_write,
        reasoning,
        notes: None,
    })
}

fn pricing_from_modelsdev_cost(cost: &ModelsDevCost) -> Option<Pricing> {
    pricing_from_rates(
        cost.input,
        cost.output,
        cost.cache_read,
        cost.cache_write,
        cost.reasoning,
    )
}

/// The `context` tiers of a cost row, ascending.
///
/// `None` — meaning "do not price this model from this feed at all" — when the
/// row carries a tier this parser cannot interpret: an unknown bound type, or
/// a band with no usable rates. A tier exists precisely because the base rates
/// stop applying above its threshold, so a tier that cannot be understood
/// cannot be safely ignored — keeping the base rates would price every
/// long-context call at a fraction of its true cost. Dropping the row instead
/// lets pricing fall back to another source, or to the documented
/// "cost unknown" fail-open.
fn tiers_from_modelsdev_cost(cost: &ModelsDevCost) -> Option<Vec<PricingTier>> {
    let base = pricing_from_modelsdev_cost(cost);
    cost.tiers
        .iter()
        .map(|tier| {
            let bound = tier.tier.as_ref()?;
            if bound.bound_type.as_deref() != Some("context") {
                return None;
            }
            // A context tier restates rate-per-token bands; the feed does not
            // publish a per-tier reasoning rate, so the base row's carries over
            // through `merge_band_over_base`.
            let band = pricing_from_rates(
                tier.input,
                tier.output,
                tier.cache_read,
                tier.cache_write,
                None,
            )?;
            Some(PricingTier {
                // The bound is inclusive: the models.dev SDK schema documents
                // `tier.size` as "Context size (in tokens) at which this tier
                // starts to apply" / "Pricing that applies from a given context
                // size upward". (The `context_over_200k` field is a legacy
                // projection the same schema marks "Prefer `tiers`", not the
                // definition — an earlier reading that took it as exclusive was
                // wrong.)
                min_input_tokens: bound.size?,
                pricing: merge_band_over_base(base, band),
            })
        })
        .collect()
}

/// Parse the models.dev `api.json` body into catalog entries.
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
            // An un-interpretable tier drops the row's pricing entirely: its
            // base rates do not apply above the tier's threshold, so keeping
            // them would under-price every long-context call.
            let tiers = model
                .cost
                .as_ref()
                .map_or_else(|| Some(Vec::new()), tiers_from_modelsdev_cost);
            let (pricing, pricing_tiers) = match tiers {
                Some(tiers) => (
                    model.cost.as_ref().and_then(pricing_from_modelsdev_cost),
                    tiers,
                ),
                None => (None, Vec::new()),
            };
            let context_window = model.limit.as_ref().and_then(|limit| limit.context);
            let max_output_tokens = model.limit.and_then(|limit| limit.output);
            entries.push(CatalogEntry {
                provider: provider.clone(),
                model_id: model.id,
                context_window,
                max_output_tokens,
                pricing,
                pricing_tiers,
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
        let client = match build_feed_client() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("model-catalog feed client build failed, using default client: {e}");
                reqwest::Client::new()
            }
        };
        Self {
            client,
            url: MODELS_DEV_URL.to_owned(),
        }
    }
}

impl ModelsDevSource {
    /// Create a source pointing at the canonical models.dev endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the feed HTTP client cannot be constructed.
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: build_feed_client()?,
            url: MODELS_DEV_URL.to_owned(),
        })
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

#[cfg(test)]
mod tests {
    use super::*;

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

    fn find<'a>(
        entries: &'a [CatalogEntry],
        provider: &str,
        model: &str,
    ) -> Result<&'a CatalogEntry> {
        entries
            .iter()
            .find(|e| e.provider == provider && e.model_id == model)
            .with_context(|| format!("missing {provider}/{model}"))
    }

    #[test]
    fn parse_modelsdev_maps_pricing_limits_and_provider() -> Result<()> {
        let entries = parse_modelsdev(MODELSDEV_FIXTURE)?;
        assert_eq!(entries.len(), 3);

        let claude = find(&entries, "anthropic", "claude-sonnet-4-5")?;
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
        let gemini = find(&entries, "gemini", "gemini-2.5-pro")?;
        assert_eq!(gemini.context_window, Some(1_048_576));
        assert_eq!(gemini.max_output_tokens, Some(65_536));

        // No model is keyed under the raw `google` provider name.
        assert!(!entries.iter().any(|e| e.provider == "google"));
        Ok(())
    }

    const ZERO_COST_FIXTURE: &str = r#"{
      "openai": {
        "id": "openai",
        "name": "OpenAI",
        "models": {
          "free-model": {
            "id": "free-model",
            "cost": { "input": 0, "output": 0 }
          },
          "half-priced-model": {
            "id": "half-priced-model",
            "cost": { "input": 2, "output": 0 }
          }
        }
      }
    }"#;

    /// Mirrors the live feed's shape for a tiered model (`cost.tiers[].tier =
    /// { type: "context", size }`), plus a row whose tier is bounded by
    /// something this parser does not understand.
    const MODELSDEV_TIERS_FIXTURE: &str = r#"{
      "openai": {
        "id": "openai",
        "name": "OpenAI",
        "models": {
          "gpt-5.4": {
            "id": "gpt-5.4",
            "cost": {
              "input": 2.5,
              "output": 15,
              "cache_read": 0.25,
              "tiers": [
                {
                  "input": 5,
                  "output": 22.5,
                  "cache_read": 0.5,
                  "tier": { "type": "context", "size": 272000 }
                }
              ]
            }
          },
          "unknown-tier-model": {
            "id": "unknown-tier-model",
            "cost": {
              "input": 1,
              "output": 2,
              "tiers": [
                {
                  "input": 9,
                  "output": 9,
                  "tier": { "type": "throughput", "size": 100 }
                }
              ]
            }
          }
        }
      },
      "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
          "claude-haiku-4-5": {
            "id": "claude-haiku-4-5",
            "cost": { "input": 1, "output": 5, "cache_read": 0.1, "cache_write": 1.25 }
          }
        }
      }
    }"#;

    #[test]
    fn parse_modelsdev_reads_tiers_and_cache_write() -> Result<()> {
        let entries = parse_modelsdev(MODELSDEV_TIERS_FIXTURE)?;

        let tiered = find(&entries, "openai", "gpt-5.4")?;
        let base = tiered.pricing.context("base pricing missing")?;
        assert!((base.input.context("input")?.usd_per_million_tokens - 2.5).abs() < f64::EPSILON);
        assert_eq!(tiered.pricing_tiers.len(), 1);
        let tier = tiered.pricing_tiers[0];
        // The bound is inclusive (SDK schema: "at which this tier starts to
        // apply"), so `size` maps straight across.
        assert_eq!(tier.min_input_tokens, 272_000);
        assert!(
            (tier
                .pricing
                .output
                .context("tier output")?
                .usd_per_million_tokens
                - 22.5)
                .abs()
                < f64::EPSILON
        );

        // The cache-write rate is carried, so cache-creation tokens can be
        // billed at the premium the provider actually charges.
        let haiku = find(&entries, "anthropic", "claude-haiku-4-5")?;
        let pricing = haiku.pricing.context("pricing missing")?;
        assert!(
            (pricing
                .cache_write
                .context("cache_write")?
                .usd_per_million_tokens
                - 1.25)
                .abs()
                < f64::EPSILON
        );
        Ok(())
    }

    /// The feed's `cost.reasoning` rate is parsed onto `Pricing`. Live rows
    /// price reasoning above output (`alibaba/qwen3-32b`: output 2.8, reasoning
    /// 8.4), which the output band's `max(output, reasoning)` must reach.
    #[test]
    fn parse_modelsdev_reads_reasoning_rate() -> Result<()> {
        const REASONING_FIXTURE: &str = r#"{
          "alibaba": {
            "id": "alibaba",
            "models": {
              "qwen3-32b": {
                "id": "qwen3-32b",
                "cost": { "input": 0.7, "output": 2.8, "reasoning": 8.4 }
              }
            }
          }
        }"#;

        let entries = parse_modelsdev(REASONING_FIXTURE)?;
        let row = find(&entries, "alibaba", "qwen3-32b")?;
        let pricing = row.pricing.context("pricing missing")?;
        assert!(
            (pricing
                .reasoning
                .context("reasoning")?
                .usd_per_million_tokens
                - 8.4)
                .abs()
                < f64::EPSILON
        );
        Ok(())
    }

    /// A tier this parser cannot interpret means the base rates cannot be
    /// trusted above *some* unknown threshold, so the row carries no pricing
    /// at all rather than a base rate that would under-price long calls.
    #[test]
    fn parse_modelsdev_drops_a_row_with_an_uninterpretable_tier() -> Result<()> {
        let entries = parse_modelsdev(MODELSDEV_TIERS_FIXTURE)?;

        let unknown = find(&entries, "openai", "unknown-tier-model")?;
        assert!(unknown.pricing.is_none());
        assert!(unknown.pricing_tiers.is_empty());
        Ok(())
    }

    /// A drifted tier row must cost that row its pricing — not the whole feed.
    /// A missing field anywhere in a 3MB body would otherwise fail the parse
    /// outright, empty the catalog, and leave every feed-priced model unpriced.
    #[test]
    fn parse_modelsdev_survives_a_tier_with_a_missing_bound() -> Result<()> {
        const DRIFTED_FIXTURE: &str = r#"{
          "openai": {
            "id": "openai",
            "models": {
              "healthy-model": {
                "id": "healthy-model",
                "cost": { "input": 1, "output": 2 }
              },
              "drifted-tier-model": {
                "id": "drifted-tier-model",
                "cost": {
                  "input": 1,
                  "output": 2,
                  "tiers": [
                    { "input": 9, "output": 9, "tier": { "type": "context" } }
                  ]
                }
              },
              "bound-less-tier-model": {
                "id": "bound-less-tier-model",
                "cost": {
                  "input": 1,
                  "output": 2,
                  "tiers": [{ "input": 9, "output": 9 }]
                }
              }
            }
          }
        }"#;

        let entries = parse_modelsdev(DRIFTED_FIXTURE)?;
        assert_eq!(
            entries.len(),
            3,
            "the parse must not abort on a drifted row"
        );

        // The healthy row is priced as usual.
        let healthy = find(&entries, "openai", "healthy-model")?;
        assert!(healthy.pricing.is_some());

        // A tier with no `size`, and a tier with no bound at all, are both
        // un-interpretable: their rows drop their pricing rather than keeping
        // base rates that stop applying above an unknown threshold.
        for model in ["drifted-tier-model", "bound-less-tier-model"] {
            let drifted = find(&entries, "openai", model)?;
            assert!(drifted.pricing.is_none(), "{model} must drop its pricing");
            assert!(drifted.pricing_tiers.is_empty());
        }
        Ok(())
    }

    #[test]
    fn parse_modelsdev_drops_zero_rates() -> Result<()> {
        let entries = parse_modelsdev(ZERO_COST_FIXTURE)?;

        // Every rate is zero: the row carries no pricing at all, so it cannot
        // shadow another source with a real price.
        let free = find(&entries, "openai", "free-model")?;
        assert!(free.pricing.is_none());

        // A zero output rate is dropped while the real input rate is kept;
        // the registry declines such a row for any call that bills output.
        let half = find(&entries, "openai", "half-priced-model")?;
        let pricing = half.pricing.context("input rate must survive")?;
        assert!(
            (pricing.input.context("input")?.usd_per_million_tokens - 2.0).abs() < f64::EPSILON
        );
        assert!(pricing.output.is_none());
        Ok(())
    }
}
