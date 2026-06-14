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

fn pricing_from_modelsdev_cost(cost: &ModelsDevCost) -> Option<Pricing> {
    if cost.input.is_none() && cost.output.is_none() && cost.cache_read.is_none() {
        return None;
    }
    Some(Pricing {
        input: cost.input.map(PricePoint::new),
        output: cost.output.map(PricePoint::new),
        cached_input: cost.cache_read.map(PricePoint::new),
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
}
