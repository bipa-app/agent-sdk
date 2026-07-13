use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;

use super::{CatalogEntry, MODELS_DEV_URL, ModelCatalogSource, build_feed_client};
use crate::model_capabilities::{PricePoint, Pricing};

#[derive(serde::Deserialize)]
struct ModelsDevCost {
    #[serde(default)]
    input: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
    #[serde(default)]
    cache_read: Option<f64>,
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

fn pricing_from_modelsdev_cost(cost: &ModelsDevCost) -> Option<Pricing> {
    let input = cost.input.and_then(modelsdev_price_per_million);
    let output = cost.output.and_then(modelsdev_price_per_million);
    let cached_input = cost.cache_read.and_then(modelsdev_price_per_million);

    if input.is_none() && output.is_none() && cached_input.is_none() {
        return None;
    }
    Some(Pricing {
        input,
        output,
        cached_input,
        notes: None,
    })
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
            let pricing = model.cost.as_ref().and_then(pricing_from_modelsdev_cost);
            let context_window = model.limit.as_ref().and_then(|limit| limit.context);
            let max_output_tokens = model.limit.and_then(|limit| limit.output);
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
