//! Run-level token / cost budget accounting.
//!
//! The agent loop reserves no budget up front; instead it checks the
//! cumulative usage at every turn-continuation boundary (see
//! [`check_budget`]) and stops the run before starting another turn once a
//! configured [`UsageLimits`] threshold is crossed.
//!
//! Cost is estimated from the run's provider/model pricing, resolved through
//! the optional [`CostEstimator`] the loop was built with (a dynamic catalog
//! such as [`ModelRegistry`](agent_sdk_providers::ModelRegistry), whose feed
//! pricing is fresher than anything compiled in) and then the static
//! [`model_capabilities`](crate::model_capabilities) table. A model priced by
//! neither reports `None` and never trips the cost limit.

use crate::pricing::CostEstimator;
use crate::types::{BudgetLimitKind, TokenUsage, UsageLimits};
use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::llm::Usage;

/// Estimate the USD cost of `usage` for the run's provider/model.
///
/// Returns `None` when no pricing source knows the provider/model pair, so
/// callers treat an un-priced model as "cost unknown" rather than free.
#[must_use]
pub(super) fn estimate_cost_usd(
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    usage: &TokenUsage,
) -> Option<f64> {
    let usage = Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
    };
    price_candidates(&provenance.provider, &provenance.model)
        .find_map(|(provider, model)| price_call(pricing, provider, model, &usage))
}

/// Price one call for a single provider/model key: the configured estimator
/// first, the static capability table as the fallback.
///
/// An estimator that holds no pricing for the pair must not make the call
/// look free, so a `None` from it falls through to the compiled-in table
/// rather than short-circuiting the lookup.
fn price_call(
    pricing: Option<&dyn CostEstimator>,
    provider: &str,
    model: &str,
    usage: &Usage,
) -> Option<f64> {
    pricing
        .and_then(|estimator| estimator.estimate_cost_usd(provider, model, usage))
        .or_else(|| {
            crate::model_capabilities::get_model_capabilities(provider, model)?
                .estimate_cost_usd(usage)
        })
}

/// The provider/model keys to price a provenance under, most specific first:
/// the pair the provider reported, then the canonical backend(s) that
/// provider name is known to serve, then — for a vendor-slug model id — the
/// vendor split out of the id.
///
/// Pricing sources key entries under the canonical provider names
/// (`anthropic` / `openai` / `gemini`), but several [`crate::llm::LlmProvider`]
/// implementations report a transport-specific name in their provenance:
/// `openai-responses` and `openai-codex` serve `openai` models, `vertex`
/// serves `anthropic` (`claude-*`) or `gemini` models, and
/// `cloudflare-ai-gateway` wraps an anthropic / openai / gemini backend
/// chosen at construction. Without the alias step those runs would silently
/// report `None` cost and [`crate::types::UsageLimits::max_cost_usd`] could
/// never trip.
///
/// The single-catalog aliases mirror the `LlmProvider::capabilities` default
/// in `agent-sdk-providers`; the gateway arm mirrors
/// `CloudflareAIGatewayProvider::capabilities`, which delegates to the
/// wrapped backend. From the provenance alone only the model remains, so the
/// gateway lookup scans the three backend catalogs in the gateway's own
/// `Inner` order (anthropic, openai, gemini) — model ids are disjoint across
/// catalogs, so the first hit is the wrapped backend's entry.
///
/// `record-replay` (`RecordReplayProvider`) wraps an arbitrary inner
/// provider while preserving the inner model id, so it resolves the same
/// way. Record mode forwards live billable calls — without the alias,
/// `max_cost_usd` could never trip on a recorded run. Replay mode bills
/// nothing but reproduces the recorded responses (and their usage)
/// deterministically; pricing it identically is deliberate, so replayed
/// runs exercise the same budget behavior the recording did — the
/// provenance string does not distinguish the modes.
fn price_candidates<'a>(
    provider: &'a str,
    model: &'a str,
) -> impl Iterator<Item = (&'a str, &'a str)> {
    let backends: &'static [&'static str] = match provider {
        "openai-responses" | "openai-codex" => &["openai"],
        "vertex" if model.starts_with("claude-") => &["anthropic"],
        "vertex" => &["gemini"],
        "cloudflare-ai-gateway" | "record-replay" => &["anthropic", "openai", "gemini"],
        _ => &[],
    };
    std::iter::once((provider, model))
        .chain(backends.iter().map(move |backend| (*backend, model)))
        .chain(vendor_slug_key(model))
}

/// Split a vendor-slug model id (`z-ai/glm-5.1`) into the provider/model key
/// the dynamic catalogs store it under (`z-ai` / `glm-5.1`).
///
/// Open models (z.ai, Moonshot, `DeepSeek`, `MiniMax`, …) are served through
/// `OpenAIProvider`, so their provenance is `openai` plus the full slug the
/// caller passed. The feeds key the same model by its vendor: `OpenRouter`
/// splits the slug (mirrored here, `google` included, so the halves match
/// `split_openrouter_id`), and models.dev nests the model under the vendor's
/// provider object. Without this key a feed-only slug model would price to
/// `None` and never trip a cost budget.
fn vendor_slug_key(model: &str) -> Option<(&str, &str)> {
    let (vendor, model_id) = model.split_once('/')?;
    let provider = if vendor == "google" { "gemini" } else { vendor };
    Some((provider, model_id))
}

/// Whether a usage delta carries no tokens at all.
#[must_use]
pub(super) const fn usage_is_zero(usage: &TokenUsage) -> bool {
    usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_input_tokens == 0
        && usage.cache_creation_input_tokens == 0
}

/// Fold one LLM call's cost into the state's accumulated total, priced at
/// the provenance that served the call.
///
/// `pre_delta_total` is the thread's aggregate usage BEFORE `delta` was
/// added; when the state predates cost accumulation (`accumulated_cost_usd`
/// is `None` while usage already exists) it is used to seed the accumulator
/// once by repricing the aggregate at the current provenance's rates — a
/// documented best-effort for legacy snapshots (see
/// [`crate::types::AgentState::accumulated_cost_usd`]). Un-priced deltas
/// (no pricing metadata) contribute nothing and never turn the accumulator
/// `Some` on their own.
pub(super) fn accumulate_cost(
    state: &mut crate::types::AgentState,
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    pre_delta_total: &TokenUsage,
    delta: &TokenUsage,
) {
    if state.accumulated_cost_usd.is_none() && !usage_is_zero(pre_delta_total) {
        state.accumulated_cost_usd = estimate_cost_usd(pricing, provenance, pre_delta_total);
    }
    if let Some(delta_cost) = estimate_cost_usd(pricing, provenance, delta) {
        state.accumulated_cost_usd = Some(state.accumulated_cost_usd.unwrap_or(0.0) + delta_cost);
    }
}

/// The run's estimated cost so far: the per-call accumulated total when
/// tracked, falling back to repricing the aggregate usage at the current
/// provenance (legacy snapshots / states that never saw a priced call —
/// best-effort, may misprice history across model rotations).
#[must_use]
pub(super) fn run_cost_usd(
    accumulated: Option<f64>,
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    usage: &TokenUsage,
) -> Option<f64> {
    accumulated.or_else(|| estimate_cost_usd(pricing, provenance, usage))
}

/// Evaluate the run-level usage budget against the cumulative `usage`.
///
/// Returns `Some((limit, estimated_cost))` when a configured limit has been
/// exceeded — the cost is carried alongside so terminal events / states can
/// report it — and `None` when budgeting is disabled or the run is still
/// within budget. `accumulated_cost_usd` is the state's per-call cost
/// accumulator (see [`crate::types::AgentState::accumulated_cost_usd`]);
/// when absent the aggregate is repriced at the current provenance as a
/// best-effort fallback for legacy snapshots.
#[must_use]
pub(super) fn status(
    usage_limits: Option<&UsageLimits>,
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    usage: &TokenUsage,
    accumulated_cost_usd: Option<f64>,
) -> Option<(BudgetLimitKind, Option<f64>)> {
    let limits = usage_limits?;
    let cost = run_cost_usd(accumulated_cost_usd, pricing, provenance, usage);
    let limit = check_budget(limits, usage, cost)?;
    Some((limit, cost))
}

/// Total billable tokens for budgeting: input + output summed.
///
/// Cache-creation / cache-read counts are already components of
/// `input_tokens` for the providers the SDK supports, so they are not added
/// again here.
#[must_use]
fn total_tokens(usage: &TokenUsage) -> u64 {
    u64::from(usage.input_tokens).saturating_add(u64::from(usage.output_tokens))
}

/// Decide whether the cumulative `usage` (and optional `cost`) has crossed
/// any configured limit.
///
/// Returns the [`BudgetLimitKind`] of the first limit exceeded, or `None`
/// when the run is still within budget. The token limit is checked before
/// the cost limit so a run that trips both reports the token limit.
#[must_use]
pub(super) fn check_budget(
    limits: &UsageLimits,
    usage: &TokenUsage,
    cost: Option<f64>,
) -> Option<BudgetLimitKind> {
    if let Some(max_total_tokens) = limits.max_total_tokens
        && total_tokens(usage) > max_total_tokens
    {
        return Some(BudgetLimitKind::TotalTokens);
    }
    if let (Some(max_cost_usd), Some(cost)) = (limits.max_cost_usd, cost)
        && cost > max_cost_usd
    {
        return Some(BudgetLimitKind::CostUsd);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentState, ThreadId};

    /// A model the static capability table has never heard of, priced at
    /// $10 / 1M input and $20 / 1M output — the shape of a model that only
    /// the dynamic catalog's feed carries.
    const FEED_MODEL: &str = "feed-only-model";

    /// Stands in for a dynamic catalog: prices exactly one canonical
    /// provider/model pair and reports `None` for everything else.
    struct FeedPricing;

    impl CostEstimator for FeedPricing {
        fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64> {
            (provider == "openai" && model == FEED_MODEL).then(|| {
                (f64::from(usage.input_tokens) / 1_000_000.0)
                    .mul_add(10.0, f64::from(usage.output_tokens) / 1_000_000.0 * 20.0)
            })
        }
    }

    /// A catalog that knows nothing: every lookup must fall through to the
    /// static table.
    struct EmptyPricing;

    impl CostEstimator for EmptyPricing {
        fn estimate_cost_usd(&self, _provider: &str, _model: &str, _usage: &Usage) -> Option<f64> {
            None
        }
    }

    fn one_million_each() -> TokenUsage {
        TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        }
    }

    #[test]
    fn feed_only_model_is_unpriced_without_a_cost_estimator() {
        let provenance = AuditProvenance::new("openai", FEED_MODEL);
        assert!(estimate_cost_usd(None, &provenance, &one_million_each()).is_none());
    }

    #[test]
    fn feed_only_model_accrues_cost_through_the_estimator() -> anyhow::Result<()> {
        use anyhow::Context;
        let provenance = AuditProvenance::new("openai", FEED_MODEL);
        let cost = estimate_cost_usd(Some(&FeedPricing), &provenance, &one_million_each())
            .context("the estimator prices this model")?;
        assert!((cost - 30.0).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    #[test]
    fn feed_only_model_trips_the_cost_limit() -> anyhow::Result<()> {
        use anyhow::Context;
        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        let provenance = AuditProvenance::new("openai", FEED_MODEL);
        let usage = one_million_each();

        // Without a catalog the model is un-priced: documented fail-open.
        assert!(status(Some(&limits), None, &provenance, &usage, None).is_none());

        let (limit, cost) = status(Some(&limits), Some(&FeedPricing), &provenance, &usage, None)
            .context("feed pricing must trip the cost limit")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        let cost = cost.context("the tripped limit carries the estimate")?;
        assert!((cost - 30.0).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    /// Stands in for a catalog carrying a vendor-slug open model under the
    /// key the feeds use: the vendor as the provider, the rest of the slug as
    /// the model id. Prices every token at $1.
    struct SlugPricing;

    impl CostEstimator for SlugPricing {
        fn estimate_cost_usd(&self, provider: &str, model: &str, usage: &Usage) -> Option<f64> {
            (provider == "z-ai" && model == "glm-9-turbo")
                .then(|| f64::from(usage.input_tokens) + f64::from(usage.output_tokens))
        }
    }

    #[test]
    fn estimator_prices_a_vendor_slug_model_under_its_feed_key() -> anyhow::Result<()> {
        use anyhow::Context;
        // Open models route through the OpenAI provider, so the provenance is
        // `openai` plus the full slug, while the feeds key the model by vendor.
        let provenance = AuditProvenance::new("openai", "z-ai/glm-9-turbo");
        let usage = TokenUsage {
            input_tokens: 20,
            output_tokens: 10,
            ..Default::default()
        };
        let cost = estimate_cost_usd(Some(&SlugPricing), &provenance, &usage)
            .context("the slug must resolve to the catalog's vendor/model key")?;
        assert!((cost - 30.0).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    #[test]
    fn estimator_prices_an_aliased_provenance() -> anyhow::Result<()> {
        use anyhow::Context;
        // `openai-codex` serves `openai` models: the estimator is consulted
        // under the canonical backend name, exactly like the static table.
        let provenance = AuditProvenance::new("openai-codex", FEED_MODEL);
        let cost = estimate_cost_usd(Some(&FeedPricing), &provenance, &one_million_each())
            .context("the aliased provenance must resolve to the openai entry")?;
        assert!((cost - 30.0).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    #[test]
    fn static_table_still_prices_what_the_estimator_misses() -> anyhow::Result<()> {
        use anyhow::Context;
        let provenance = AuditProvenance::new("openai", "gpt-4o");
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let with_empty_catalog = estimate_cost_usd(Some(&EmptyPricing), &provenance, &usage)
            .context("an estimator miss must fall back to the static table")?;
        let statically_priced = estimate_cost_usd(None, &provenance, &usage)
            .context("gpt-4o is priced in the static table")?;
        assert!((with_empty_catalog - statically_priced).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn accumulate_cost_uses_the_estimator_for_feed_priced_calls() -> anyhow::Result<()> {
        use anyhow::Context;
        let provenance = AuditProvenance::new("openai", FEED_MODEL);
        let mut state = AgentState::new(ThreadId::new());

        accumulate_cost(
            &mut state,
            Some(&FeedPricing),
            &provenance,
            &TokenUsage::default(),
            &one_million_each(),
        );

        let accumulated = state
            .accumulated_cost_usd
            .context("a feed-priced call must accumulate cost")?;
        assert!(
            (accumulated - 30.0).abs() < 1e-9,
            "unexpected accumulated cost: {accumulated}"
        );
        Ok(())
    }

    #[test]
    fn estimate_cost_for_gpt_4o_matches_pricing() -> anyhow::Result<()> {
        use anyhow::Context;
        // gpt-4o: $1.25 / 1M input, $5.00 / 1M output.
        // 2000 input + 1000 output = 0.0025 + 0.005 = 0.0075 USD.
        let provenance = AuditProvenance::new("openai", "gpt-4o");
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let cost = estimate_cost_usd(None, &provenance, &usage).context("gpt-4o has pricing")?;
        assert!((cost - 0.0075).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_openai_responses_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(None, &AuditProvenance::new("openai", "gpt-4o"), &usage)
            .context("gpt-4o has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("openai-responses", "gpt-4o"),
            &usage,
        )
        .context("openai-responses must resolve to the openai catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_openai_codex_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(None, &AuditProvenance::new("openai", "gpt-4o"), &usage)
            .context("gpt-4o has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("openai-codex", "gpt-4o"),
            &usage,
        )
        .context("openai-codex must resolve to the openai catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_vertex_claude_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(
            None,
            &AuditProvenance::new("anthropic", "claude-fable-5"),
            &usage,
        )
        .context("claude-fable-5 has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("vertex", "claude-fable-5"),
            &usage,
        )
        .context("vertex claude-* must resolve to the anthropic catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_vertex_gemini_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(
            None,
            &AuditProvenance::new("gemini", "gemini-3.1-pro"),
            &usage,
        )
        .context("gemini-3.1-pro has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("vertex", "gemini-3.1-pro"),
            &usage,
        )
        .context("vertex non-claude must resolve to the gemini catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_cloudflare_gateway_claude_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(
            None,
            &AuditProvenance::new("anthropic", "claude-fable-5"),
            &usage,
        )
        .context("claude-fable-5 has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("cloudflare-ai-gateway", "claude-fable-5"),
            &usage,
        )
        .context("gateway claude-* must resolve to the anthropic catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_cloudflare_gateway_openai_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(None, &AuditProvenance::new("openai", "gpt-4o"), &usage)
            .context("gpt-4o has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("cloudflare-ai-gateway", "gpt-4o"),
            &usage,
        )
        .context("gateway gpt-* must resolve to the openai catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_record_replay_anthropic_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(
            None,
            &AuditProvenance::new("anthropic", "claude-fable-5"),
            &usage,
        )
        .context("claude-fable-5 has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("record-replay", "claude-fable-5"),
            &usage,
        )
        .context("record-replay claude-* must resolve to the anthropic catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_record_replay_openai_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(None, &AuditProvenance::new("openai", "gpt-4o"), &usage)
            .context("gpt-4o has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("record-replay", "gpt-4o"),
            &usage,
        )
        .context("record-replay gpt-* must resolve to the openai catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_resolves_record_replay_gemini_alias() -> anyhow::Result<()> {
        use anyhow::Context;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let canonical = estimate_cost_usd(
            None,
            &AuditProvenance::new("gemini", "gemini-3.1-pro"),
            &usage,
        )
        .context("gemini-3.1-pro has pricing")?;
        let aliased = estimate_cost_usd(
            None,
            &AuditProvenance::new("record-replay", "gemini-3.1-pro"),
            &usage,
        )
        .context("record-replay gemini-* must resolve to the gemini catalog entry")?;
        assert!((aliased - canonical).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn estimate_cost_is_none_for_unknown_model() {
        let provenance = AuditProvenance::new("mock", "mock-model");
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 100,
            ..Default::default()
        };
        assert!(estimate_cost_usd(None, &provenance, &usage).is_none());
    }

    #[test]
    fn token_budget_trips_when_exceeded() {
        let limits = UsageLimits {
            max_total_tokens: Some(100),
            ..Default::default()
        };
        let under = TokenUsage {
            input_tokens: 50,
            output_tokens: 50,
            ..Default::default()
        };
        assert!(check_budget(&limits, &under, None).is_none());

        let over = TokenUsage {
            input_tokens: 60,
            output_tokens: 60,
            ..Default::default()
        };
        assert_eq!(
            check_budget(&limits, &over, None),
            Some(BudgetLimitKind::TotalTokens)
        );
    }

    #[test]
    fn cost_budget_trips_when_exceeded() {
        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        let usage = TokenUsage::default();
        assert!(check_budget(&limits, &usage, Some(0.5)).is_none());
        assert_eq!(
            check_budget(&limits, &usage, Some(1.5)),
            Some(BudgetLimitKind::CostUsd)
        );
        // Unknown cost (None) never trips the cost limit.
        assert!(check_budget(&limits, &usage, None).is_none());
    }
}

/// Budget pricing driven by a real [`ModelRegistry`] loaded from a feed body,
/// so the provider/model keys under test are the ones the feed parsers
/// actually produce.
#[cfg(all(test, feature = "model-discovery"))]
mod catalog_tests {
    use super::*;
    use agent_sdk_providers::model_catalog::parse_openrouter;
    use agent_sdk_providers::{CatalogEntry, ModelCatalogSource, ModelRegistry};
    use anyhow::{Context, Result};
    use async_trait::async_trait;

    /// A vendor-slug open model, priced only by the feed: `OpenAIProvider`
    /// serves it, so the run's provenance is `openai` / `z-ai/glm-9-turbo`,
    /// while the feed keys it as `z-ai` / `glm-9-turbo`.
    const OPENROUTER_SLUG_FIXTURE: &str = r#"{
      "data": [
        {
          "id": "z-ai/glm-9-turbo",
          "context_length": 200000,
          "pricing": { "prompt": "0.000001", "completion": "0.000002" }
        }
      ]
    }"#;

    /// A feed row for a model the static table also prices (`openai/gpt-4o`),
    /// but carrying only an input rate.
    const OPENROUTER_PARTIAL_FIXTURE: &str = r#"{
      "data": [
        {
          "id": "openai/gpt-4o",
          "context_length": 128000,
          "pricing": { "prompt": "0.001" }
        }
      ]
    }"#;

    struct FixtureFeed(Vec<CatalogEntry>);

    #[async_trait]
    impl ModelCatalogSource for FixtureFeed {
        async fn fetch(&self) -> Result<Vec<CatalogEntry>> {
            Ok(self.0.clone())
        }
    }

    async fn registry_from(fixture: &str) -> Result<ModelRegistry> {
        let registry = ModelRegistry::new();
        registry
            .refresh(&FixtureFeed(parse_openrouter(fixture)?))
            .await?;
        Ok(registry)
    }

    #[tokio::test]
    async fn feed_only_slug_model_accrues_cost_and_trips_the_budget() -> Result<()> {
        let registry = registry_from(OPENROUTER_SLUG_FIXTURE).await?;
        // $1 / 1M input and $2 / 1M output, over 1M tokens each: $3.
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let provenance = AuditProvenance::new("openai", "z-ai/glm-9-turbo");

        // The static table has never heard of this model.
        assert!(estimate_cost_usd(None, &provenance, &usage).is_none());

        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the feed prices this slug model")?;
        assert!((cost - 3.0).abs() < 1e-9, "unexpected cost: {cost}");

        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        let (limit, _) = status(Some(&limits), Some(&registry), &provenance, &usage, None)
            .context("feed pricing must trip the cost limit")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    #[tokio::test]
    async fn partially_priced_feed_row_yields_to_the_static_table() -> Result<()> {
        let registry = registry_from(OPENROUTER_PARTIAL_FIXTURE).await?;
        let usage = TokenUsage {
            input_tokens: 2_000,
            output_tokens: 1_000,
            ..Default::default()
        };
        let provenance = AuditProvenance::new("openai", "gpt-4o");

        // The feed row prices input at $1000/1M and says nothing about
        // output; pricing from it would bill 1000 output tokens at zero. The
        // catalog declines, so the fully-priced static row prices the call.
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the static table prices gpt-4o")?;
        let statically_priced = estimate_cost_usd(None, &provenance, &usage)
            .context("gpt-4o is priced in the static table")?;
        assert!((cost - statically_priced).abs() < 1e-12);
        assert!((cost - 0.0075).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }
}
