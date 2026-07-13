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
//! [`AgentLoopBuilder::cost_estimator`](crate::AgentLoopBuilder::cost_estimator);
//! the loop consults it first for every priced call and falls back to the
//! static table whenever the estimator has no price for the provider/model
//! pair.
//!
//! The `model-discovery` feature implements the trait for
//! [`ModelRegistry`](agent_sdk_providers::ModelRegistry), the layered
//! (override → third-party feed → static table) catalog, so a run can price —
//! and budget — models whose pricing only ever appeared in the feed.

use crate::llm::Usage;

/// A source of provider/model pricing for run-level cost budgeting.
///
/// Implementations answer for a single provider/model pair at a time and
/// report `None` when they hold no pricing for it, which lets the loop fall
/// back to the static capability table rather than treating the call as free.
pub trait CostEstimator: Send + Sync {
    /// Estimate the USD cost of `usage` for `provider`/`model`.
    ///
    /// `provider` is the canonical provider name (`anthropic`, `openai`,
    /// `gemini`, …): the loop normalizes transport-specific provenance names
    /// (`openai-responses`, `vertex`, `cloudflare-ai-gateway`, …) to the
    /// backends they serve before calling this, so an implementation only
    /// needs to key on canonical names.
    ///
    /// Returning `None` means "no pricing for this pair" — not "free".
    fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64>;
}

#[cfg(feature = "model-discovery")]
impl CostEstimator for agent_sdk_providers::ModelRegistry {
    fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64> {
        // Resolves override → feed → static table; the inherent method, not
        // this trait method, so the layered lookup runs rather than recursing.
        Self::estimate_cost_usd(self, provider, model, usage)
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
