//! Pricing seam for run-level cost budgets.
//!
//! [`UsageLimits::max_cost_usd`](crate::types::UsageLimits::max_cost_usd)
//! stops a run once its estimated spend crosses a threshold, which requires
//! turning token counts into dollars. Out of the box the SDK prices from the
//! static [`model_capabilities`](crate::model_capabilities) table compiled
//! into the binary: a model that table has never heard of estimates to `None`
//! ("cost unknown"), accrues nothing, and can never trip the limit.
//!
//! A [`CostEstimator`] plugs a richer pricing source in front of that table.
//! Wire one in with
//! [`AgentLoopBuilder::cost_estimator`](crate::AgentLoopBuilder::cost_estimator).
//! The loop asks the estimator about every key the call could be filed under
//! before it asks the static table about any of them, so a fresher feed price
//! wins over a compiled-in rate even for a model both sources know.
//!
//! The `model-discovery` feature implements the trait for
//! [`ModelRegistry`](agent_sdk_providers::ModelRegistry), the layered catalog,
//! so a run can price — and budget — models whose pricing only ever appeared
//! in a feed.

use crate::llm::Usage;

/// A source of provider/model pricing for run-level cost budgeting.
///
/// Implementations answer for a single provider/model pair at a time and
/// report `None` when they hold no pricing of their own for it. The loop then
/// keeps looking: at the other keys the pair may be filed under, and finally
/// at the static capability table. So `None` must mean "I do not price this",
/// never "this is free" — and an implementation should not answer on the
/// static table's behalf, or the loop cannot tell the two sources apart.
pub trait CostEstimator: Send + Sync {
    /// Estimate the USD cost of `usage` for `provider`/`model`.
    ///
    /// The loop calls this once per key it believes the model could be filed
    /// under: the provenance pair as reported, the canonical backend(s) a
    /// transport-specific provider name serves (`openai-responses`, `vertex`,
    /// `cloudflare-ai-gateway`, …), and — for a vendor-slug model id — the
    /// route and vendor keys the third-party feeds use. An implementation only
    /// needs to answer for the keys it actually holds.
    ///
    /// Returning `None` means "no pricing for this pair" — not "free".
    ///
    /// `usage` describes ONE provider call, so an implementation may read its
    /// input-token count as that call's context size (e.g. to select a
    /// long-context price tier). For a summed usage the loop calls
    /// [`estimate_aggregate_cost_usd`](Self::estimate_aggregate_cost_usd)
    /// instead.
    fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64>;

    /// Estimate the USD cost of a *summed* `usage` — a thread's cumulative
    /// tokens rather than one call.
    ///
    /// The distinction matters to any source with context-dependent rates: ten
    /// 50K-token calls sum to 500K input tokens without a single call ever
    /// reaching a 272K long-context threshold, so a sum must not be read as a
    /// context size. An implementation with no such rates can ignore the
    /// distinction, which is what the default does.
    fn estimate_aggregate_cost_usd(
        &self,
        provider: &str,
        model: &str,
        usage: &Usage,
    ) -> Option<f64> {
        self.estimate_cost_usd(provider, model, usage)
    }
}

#[cfg(feature = "model-discovery")]
impl CostEstimator for agent_sdk_providers::ModelRegistry {
    fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64> {
        // The dynamic layers only (override → feed). The registry's layered
        // `estimate_cost_usd` would also answer from the static table, and a
        // static hit on the first key tried would then pre-empt the feed's
        // price for a key derived later — the caller applies the static table
        // itself, after this source has been offered every key.
        Self::estimate_dynamic_cost_usd(self, provider, model, usage)
    }

    fn estimate_aggregate_cost_usd(
        &self,
        provider: &str,
        model: &str,
        usage: &Usage,
    ) -> Option<f64> {
        Self::estimate_dynamic_base_cost_usd(self, provider, model, usage)
    }
}

#[cfg(all(test, feature = "model-discovery"))]
mod tests {
    use super::*;
    use agent_sdk_providers::{CatalogEntry, ModelRegistry};
    use anyhow::Context;

    #[test]
    fn model_registry_prices_through_the_trait() -> anyhow::Result<()> {
        let registry = ModelRegistry::new().with_override(
            "openai",
            "feed-only-model",
            CatalogEntry {
                provider: "openai".to_owned(),
                model_id: "feed-only-model".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(crate::model_capabilities::Pricing::flat(10.0, 20.0)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            },
        );
        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };

        let estimator: &dyn CostEstimator = &registry;
        let cost = estimator
            .estimate_cost_usd("openai", "feed-only-model", &usage)
            .context("registry must price a model it carries")?;
        assert!((cost - 30.0).abs() < 1e-9, "unexpected cost: {cost}");

        assert!(
            estimator
                .estimate_cost_usd("openai", "no-such-model", &usage)
                .is_none()
        );
        Ok(())
    }
}
