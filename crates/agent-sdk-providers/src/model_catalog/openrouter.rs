use anyhow::{Context, Result};
use async_trait::async_trait;

use super::{
    CatalogEntry, ModelCatalogSource, OPENROUTER_URL, PricingTier, build_feed_client,
    merge_band_over_base,
};
use crate::model_capabilities::{PricePoint, Pricing};

#[derive(serde::Deserialize)]
struct OpenRouterPricing {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
    /// The default (5-minute) cache-write rate. The feed also publishes
    /// `input_cache_write_1h` for Anthropic's extended TTL, which the SDK does
    /// not request.
    #[serde(default)]
    input_cache_write: Option<String>,
    /// Long-context price bands: from `min_prompt_tokens` upwards the route
    /// bills at these rates instead. Gemini 2.5 Pro doubles its input rate
    /// above 200K tokens this way, GPT-5.x above 272K.
    #[serde(default)]
    overrides: Vec<OpenRouterPricingOverride>,
}

/// Every field is optional: a drifted override must cost its own row's pricing,
/// never the parse of the whole feed body. See [`tiers_from_openrouter_pricing`].
#[derive(serde::Deserialize)]
struct OpenRouterPricingOverride {
    #[serde(default)]
    min_prompt_tokens: Option<u32>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
    #[serde(default)]
    input_cache_write: Option<String>,
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
            let base = model.pricing.as_ref().and_then(base_pricing);
            // An un-interpretable override drops the row's pricing entirely:
            // its base rates stop applying somewhere the parser cannot locate,
            // so keeping them would under-price every long-context call.
            let tiers = model.pricing.as_ref().map_or_else(
                || Some(Vec::new()),
                |p| tiers_from_openrouter_pricing(p, base),
            );
            let (pricing, pricing_tiers) =
                tiers.map_or_else(|| (None, Vec::new()), |tiers| (base, tiers));
            let max_output_tokens = model.top_provider.and_then(|tp| tp.max_completion_tokens);
            CatalogEntry {
                provider: OPENROUTER_PROVIDER.to_owned(),
                model_id: model.id,
                context_window: model.context_length,
                max_output_tokens,
                pricing,
                pricing_tiers,
                supports_thinking: None,
            }
        })
        .collect())
}

/// The route's rates for a call inside the base band.
fn base_pricing(pricing: &OpenRouterPricing) -> Option<Pricing> {
    pricing_from_rates(
        pricing.prompt.as_deref(),
        pricing.completion.as_deref(),
        pricing.input_cache_read.as_deref(),
        pricing.input_cache_write.as_deref(),
    )
}

fn pricing_from_rates(
    prompt: Option<&str>,
    completion: Option<&str>,
    input_cache_read: Option<&str>,
    input_cache_write: Option<&str>,
) -> Option<Pricing> {
    let input = prompt.and_then(openrouter_price_per_million);
    let output = completion.and_then(openrouter_price_per_million);
    let cached_input = input_cache_read.and_then(openrouter_price_per_million);
    let cache_write = input_cache_write.and_then(openrouter_price_per_million);

    if input.is_none() && output.is_none() && cached_input.is_none() && cache_write.is_none() {
        return None;
    }
    Some(Pricing {
        input,
        output,
        cached_input,
        cache_write,
        notes: None,
    })
}

/// The route's long-context bands, ascending.
///
/// `None` — "do not price this route from this feed at all" — when an override
/// cannot be interpreted: no `min_prompt_tokens` to locate the band, or no
/// input/output rate to bill it with. An override exists precisely because the
/// base rates stop applying above its threshold, so one that cannot be read
/// cannot be ignored: keeping the base rates would price every long-context
/// call at a fraction of its true cost.
///
/// An override that restates only *some* rates (six routes raise input, output
/// and cache-read above the threshold while leaving cache-write unstated) keeps
/// its base value for the rest — see [`merge_band_over_base`].
///
/// # Threshold and precedence
///
/// The bound is inclusive: `OpenRouter`'s provider docs state a tier "applies
/// when input tokens meet or exceed the `min_context` value", so
/// `min_prompt_tokens` maps directly to the inclusive
/// [`PricingTier::min_input_tokens`] with no offset.
///
/// The docs define pricing only for the base band plus a single override tier;
/// how *multiple* overrides compose is undocumented. `OpenRouter` applies the
/// override list in source order, so this reads "later entry wins per price
/// key": for a call, every override whose threshold the call meets is folded
/// over the base in list order, a later entry's stated rates overriding an
/// earlier one's. That fold is precomputed per distinct threshold here, so the
/// highest-threshold [`PricingTier`] a call reaches already carries the
/// resolved rates — matching what `applicable_pricing` selects. Ascending
/// lists (every live route) are unaffected; only a non-monotonic list (which
/// only a trimmed mirror could produce) changes, and there the source-order
/// reading is the conservative one.
fn tiers_from_openrouter_pricing(
    pricing: &OpenRouterPricing,
    base: Option<Pricing>,
) -> Option<Vec<PricingTier>> {
    let mut bands: Vec<(u32, Pricing)> = Vec::with_capacity(pricing.overrides.len());
    for band in &pricing.overrides {
        let rates = pricing_from_rates(
            band.prompt.as_deref(),
            band.completion.as_deref(),
            band.input_cache_read.as_deref(),
            band.input_cache_write.as_deref(),
        )?;
        // A band that bills only a cache component, with no input or output
        // rate of its own, is not a price band this parser can stand behind.
        if rates.input.is_none() || rates.output.is_none() {
            return None;
        }
        // `min_prompt_tokens` is the smallest prompt the band applies to, so
        // the bound is already inclusive.
        bands.push((band.min_prompt_tokens?, rates));
    }

    let mut thresholds: Vec<u32> = bands.iter().map(|(threshold, _)| *threshold).collect();
    thresholds.sort_unstable();
    thresholds.dedup();

    Some(
        thresholds
            .into_iter()
            .filter_map(|threshold| {
                // Fold every override the call would satisfy at this threshold,
                // in source order, each overriding the last per price key.
                let mut pricing = base;
                for (min, rates) in &bands {
                    if *min <= threshold {
                        pricing = Some(merge_band_over_base(pricing, *rates));
                    }
                }
                pricing.map(|pricing| PricingTier {
                    min_input_tokens: threshold,
                    pricing,
                })
            })
            .collect(),
    )
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

    /// Mirrors the live feed: `google/gemini-2.5-pro` raises prompt, completion
    /// and cache-read above 200K prompt tokens but leaves cache-write unstated,
    /// while `openai/gpt-5.6-luna` restates every rate above 272K.
    const OPENROUTER_OVERRIDES_FIXTURE: &str = r#"{
      "data": [
        {
          "id": "google/gemini-2.5-pro",
          "context_length": 1048576,
          "pricing": {
            "prompt": "0.00000125",
            "completion": "0.00001",
            "input_cache_read": "0.000000125",
            "input_cache_write": "0.000000375",
            "overrides": [
              {
                "min_prompt_tokens": 200000,
                "prompt": "0.0000025",
                "completion": "0.000015",
                "input_cache_read": "0.00000025"
              }
            ]
          }
        },
        {
          "id": "openai/gpt-5.6-luna",
          "context_length": 400000,
          "pricing": {
            "prompt": "0.000001",
            "completion": "0.000006",
            "input_cache_read": "0.0000001",
            "input_cache_write": "0.00000125",
            "overrides": [
              {
                "min_prompt_tokens": 272000,
                "prompt": "0.000002",
                "completion": "0.000009",
                "input_cache_read": "0.0000002",
                "input_cache_write": "0.0000025"
              }
            ]
          }
        }
      ]
    }"#;

    #[test]
    fn parse_openrouter_reads_long_context_overrides() -> Result<()> {
        let entries = parse_openrouter(OPENROUTER_OVERRIDES_FIXTURE)?;

        let gemini = find(&entries, "openrouter", "google/gemini-2.5-pro")?;
        assert_eq!(gemini.pricing_tiers.len(), 1);
        let band = gemini.pricing_tiers[0];
        // `min_prompt_tokens` is already an inclusive lower bound.
        assert_eq!(band.min_input_tokens, 200_000);
        assert!(
            (band.pricing.input.context("input")?.usd_per_million_tokens - 2.5).abs()
                < f64::EPSILON
        );
        assert!(
            (band
                .pricing
                .output
                .context("output")?
                .usd_per_million_tokens
                - 15.0)
                .abs()
                < f64::EPSILON
        );
        // The override says nothing about cache-write, so the band keeps the
        // base rate for it rather than dropping it or re-billing it at input.
        assert!(
            (band
                .pricing
                .cache_write
                .context("cache_write inherited from base")?
                .usd_per_million_tokens
                - 0.375)
                .abs()
                < f64::EPSILON
        );
        Ok(())
    }

    #[tokio::test]
    async fn long_context_route_bills_at_the_override_rate() -> Result<()> {
        let registry = ModelRegistry::new();
        registry
            .refresh(&StaticSource(parse_openrouter(
                OPENROUTER_OVERRIDES_FIXTURE,
            )?))
            .await?;

        // Inside the base band: 100K in + 100K out = 0.1*1.25 + 0.1*10 = 1.125.
        let short = Usage {
            input_tokens: 100_000,
            output_tokens: 100_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        let short_cost = registry
            .estimate_cost_usd("openrouter", "google/gemini-2.5-pro", &short)
            .context("cost estimate missing")?;
        assert!(
            (short_cost - 1.125).abs() < 1e-9,
            "unexpected cost: {short_cost}"
        );

        // At the threshold — the bound is inclusive — and past it: the override
        // rates apply to the whole call. 200K in + 100K out =
        // 0.2*2.5 + 0.1*15 = 2.0, where the base rates would say 1.25.
        for input_tokens in [200_000, 300_000] {
            let long = Usage {
                input_tokens,
                output_tokens: 100_000,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            };
            let cost = registry
                .estimate_cost_usd("openrouter", "google/gemini-2.5-pro", &long)
                .context("cost estimate missing")?;
            let expected = (f64::from(input_tokens) / 1_000_000.0).mul_add(2.5, 1.5);
            assert!(
                (cost - expected).abs() < 1e-9,
                "unexpected cost at {input_tokens}: {cost}"
            );
        }

        // One token below the threshold still pays base rates.
        let just_under = Usage {
            input_tokens: 199_999,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        let under_cost = registry
            .estimate_cost_usd("openrouter", "google/gemini-2.5-pro", &just_under)
            .context("cost estimate missing")?;
        assert!(
            (under_cost - 0.249_998_75).abs() < 1e-9,
            "unexpected cost: {under_cost}"
        );
        Ok(())
    }

    /// A summed usage is not a prompt size: the base band must price it, or a
    /// thread of short calls gets re-billed at a long-context rate it never
    /// paid. The tier machinery is shared, so this holds for both feeds.
    #[tokio::test]
    async fn aggregate_repricing_ignores_openrouter_overrides() -> Result<()> {
        let registry = ModelRegistry::new();
        registry
            .refresh(&StaticSource(parse_openrouter(
                OPENROUTER_OVERRIDES_FIXTURE,
            )?))
            .await?;

        // Three 100K-prompt calls: 300K summed, no call near the 200K bound.
        let aggregate = Usage {
            input_tokens: 300_000,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        let base = registry
            .estimate_dynamic_base_cost_usd("openrouter", "google/gemini-2.5-pro", &aggregate)
            .context("cost estimate missing")?;
        // Base: 0.3 * 1.25 = 0.375, not the override's 0.3 * 2.5 = 0.75.
        assert!((base - 0.375).abs() < 1e-9, "unexpected cost: {base}");
        Ok(())
    }

    /// `OpenRouter` applies overrides in source order (later wins per key). A
    /// non-monotonic list — a later entry at a *lower* threshold — must fold so
    /// that entry wins where both apply, not the highest-threshold entry.
    #[tokio::test]
    async fn non_monotonic_overrides_fold_later_wins() -> Result<()> {
        // Later entry (min 100K, prompt 3) is listed after the higher-threshold
        // entry (min 200K, prompt 5). Only a trimmed mirror produces this.
        const NON_MONOTONIC_FIXTURE: &str = r#"{
          "data": [
            {
              "id": "vendor/model",
              "pricing": {
                "prompt": "0.000001",
                "completion": "0.000001",
                "overrides": [
                  { "min_prompt_tokens": 200000, "prompt": "0.000005", "completion": "0.000005" },
                  { "min_prompt_tokens": 100000, "prompt": "0.000003", "completion": "0.000003" }
                ]
              }
            }
          ]
        }"#;

        let entries = parse_openrouter(NON_MONOTONIC_FIXTURE)?;
        let model = find(&entries, "openrouter", "vendor/model")?;
        // Two distinct thresholds → two tiers.
        assert_eq!(model.pricing_tiers.len(), 2);

        let registry = ModelRegistry::new();
        registry.refresh(&StaticSource(entries)).await?;
        let out = |n: u32| Usage {
            input_tokens: n,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };

        // Below 100K: base $1/M.
        let base = registry
            .estimate_cost_usd("openrouter", "vendor/model", &out(50_000))
            .context("cost estimate missing")?;
        assert!((base - 0.05).abs() < 1e-9, "unexpected cost: {base}");

        // 150K: only the min-100K entry applies → $3/M.
        let mid = registry
            .estimate_cost_usd("openrouter", "vendor/model", &out(150_000))
            .context("cost estimate missing")?;
        assert!((mid - 0.45).abs() < 1e-9, "unexpected cost: {mid}");

        // 250K: both apply; source order folds the min-100K entry last, so it
        // wins — $3/M, NOT the min-200K entry's $5/M that highest-threshold
        // selection would have picked.
        let high = registry
            .estimate_cost_usd("openrouter", "vendor/model", &out(250_000))
            .context("cost estimate missing")?;
        assert!((high - 0.75).abs() < 1e-9, "unexpected cost: {high}");
        Ok(())
    }

    /// A drifted override costs its own row's pricing, never the feed parse.
    #[test]
    fn parse_openrouter_survives_a_drifted_override() -> Result<()> {
        const DRIFTED_FIXTURE: &str = r#"{
          "data": [
            {
              "id": "vendor/healthy",
              "pricing": { "prompt": "0.000001", "completion": "0.000002" }
            },
            {
              "id": "vendor/no-threshold",
              "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002",
                "overrides": [{ "prompt": "0.000009", "completion": "0.000009" }]
              }
            },
            {
              "id": "vendor/no-rates",
              "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002",
                "overrides": [{ "min_prompt_tokens": 200000, "input_cache_read": "0.0000001" }]
              }
            }
          ]
        }"#;

        let entries = parse_openrouter(DRIFTED_FIXTURE)?;
        assert_eq!(
            entries.len(),
            3,
            "the parse must not abort on a drifted row"
        );

        let healthy = find(&entries, "openrouter", "vendor/healthy")?;
        assert!(healthy.pricing.is_some());
        assert!(healthy.pricing_tiers.is_empty());

        // An override with no threshold cannot be located, and one with no
        // input/output rate cannot be billed: either way the base rates provably
        // stop applying somewhere unknown, so the row carries no pricing at all.
        for model in ["vendor/no-threshold", "vendor/no-rates"] {
            let drifted = find(&entries, "openrouter", model)?;
            assert!(drifted.pricing.is_none(), "{model} must drop its pricing");
            assert!(drifted.pricing_tiers.is_empty());
        }
        Ok(())
    }
}
