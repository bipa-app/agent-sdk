use anyhow::{Context, Result};
use async_trait::async_trait;

use super::{CatalogEntry, ModelCatalogSource, OPENROUTER_URL, build_feed_client};
use crate::model_capabilities::{PricePoint, Pricing};

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

/// The provider name every `OpenRouter` row is filed under: the route is what
/// the caller pays, so the route is what the row is keyed by.
const OPENROUTER_PROVIDER: &str = "openrouter";

fn openrouter_price_per_million(value: &str) -> Option<PricePoint> {
    let per_token: f64 = value.trim().parse().ok()?;
    if !per_token.is_finite() || per_token <= 0.0 {
        return None;
    }
    Some(PricePoint::new(per_token * 1_000_000.0))
}

/// Parse the `OpenRouter` `/models` body into catalog entries.
///
/// Every row is filed under the route that serves it — provider `openrouter`,
/// model id the full slug (`anthropic/claude-opus-4.8`) — never under the
/// vendor half of the slug. The prices in this feed are `OpenRouter`'s own,
/// and they are not the vendor's: splitting `openai/gpt-4o` into
/// `("openai", "gpt-4o")` would file a router's rate under the exact key a
/// *direct* `OpenAI` call resolves to, so a direct call would be priced at the
/// router's rate. A caller that routes through `OpenRouter` reaches these rows
/// by asking for the route key; a caller that does not, never sees them.
///
/// This matches how models.dev files the same rows (under its `openrouter`
/// service key, slug intact).
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
                provider: OPENROUTER_PROVIDER.to_owned(),
                model_id: model.id,
                context_window: model.context_length,
                max_output_tokens,
                pricing,
                supports_thinking: None,
            }
        })
        .collect())
}

/// An alternative public feed: <https://openrouter.ai/api/v1/models> (no key).
pub struct OpenRouterSource {
    client: reqwest::Client,
    url: String,
}

impl Default for OpenRouterSource {
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
            url: OPENROUTER_URL.to_owned(),
        }
    }
}

impl OpenRouterSource {
    /// Create a source pointing at the canonical `OpenRouter` models endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the feed HTTP client cannot be constructed.
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: build_feed_client()?,
            url: OPENROUTER_URL.to_owned(),
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

#[cfg(test)]
mod tests {
    use super::super::ModelRegistry;
    use super::*;
    use agent_sdk_foundation::llm::Usage;

    struct StaticSource(Vec<CatalogEntry>);

    #[async_trait]
    impl ModelCatalogSource for StaticSource {
        async fn fetch(&self) -> Result<Vec<CatalogEntry>> {
            Ok(self.0.clone())
        }
    }

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
    fn parse_openrouter_converts_per_token_to_per_million_and_keys_rows_by_route() -> Result<()> {
        let entries = parse_openrouter(OPENROUTER_FIXTURE)?;
        assert_eq!(entries.len(), 2);

        // Rows are filed under the route, slug intact — never under the vendor
        // half, which is where a DIRECT call to that vendor would look and
        // would then find this router's price.
        assert!(entries.iter().all(|e| e.provider == "openrouter"));
        assert!(
            !entries
                .iter()
                .any(|e| e.provider == "anthropic" || e.provider == "gemini"),
        );

        let opus = find(&entries, "openrouter", "anthropic/claude-opus-4.8")?;
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

        let gemini = find(&entries, "openrouter", "google/gemini-2.5-pro")?;
        assert_eq!(gemini.context_window, Some(1_048_576));
        Ok(())
    }

    #[tokio::test]
    async fn parse_openrouter_treats_minus_one_sentinel_prices_as_absent() -> Result<()> {
        let entries = parse_openrouter(OPENROUTER_SENTINEL_FIXTURE)?;
        assert_eq!(entries.len(), 1);

        let auto = find(&entries, "openrouter", "openrouter/auto")?;
        assert!(
            auto.pricing.is_none(),
            "sentinel `-1` prices must yield None pricing, got {:?}",
            auto.pricing
        );
        assert_eq!(auto.context_window, Some(2_000_000));

        let registry = ModelRegistry::new();
        registry.refresh(&StaticSource(entries)).await?;
        let usage = Usage {
            input_tokens: 1_000,
            output_tokens: 1_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        assert_eq!(
            registry.estimate_cost_usd("openrouter", "openrouter/auto", &usage),
            None
        );
        Ok(())
    }
}
