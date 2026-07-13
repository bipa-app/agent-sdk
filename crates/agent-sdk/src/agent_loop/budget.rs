//! Run-level token / cost budget accounting.
//!
//! The agent loop reserves no budget up front; instead it checks the
//! cumulative usage at every turn-continuation boundary (see
//! [`check_budget`]) and stops the run before starting another turn once a
//! configured [`UsageLimits`] threshold is crossed.
//!
//! Cost is estimated from the run's provider/model pricing. Two sources can
//! answer: the optional [`CostEstimator`] the loop was built with (a dynamic
//! catalog such as [`ModelRegistry`](agent_sdk_providers::ModelRegistry),
//! whose feed pricing is fresher than anything compiled in) and the static
//! [`model_capabilities`](crate::model_capabilities) table.
//!
//! A provenance can be filed under several keys — provider aliases, and for a
//! routed model the router's own key or the vendor's — and the sources do not
//! agree on which. So the lookup is source-major: the estimator is asked about
//! every key before the static table is asked about any (see
//! [`estimate_cost_usd`]). A model priced by neither source reports `None` and
//! never trips the cost limit.

use crate::pricing::CostEstimator;
use crate::types::{BudgetLimitKind, TokenUsage, UsageLimits};
use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::llm::Usage;

/// What a [`TokenUsage`] being priced describes.
///
/// A source with context-dependent rates reads a single call's input-token
/// count as that call's context size. A thread's summed usage is not a context
/// size — ten 50K-token calls sum to 500K without any one call approaching a
/// 272K long-context threshold — so the two must be priced differently, or
/// repricing a healthy thread's history invents a long-context bill it never
/// paid and trips its budget.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum UsageScope {
    /// One provider call, priced at the rates that call actually paid.
    Call,
    /// A summed usage (a thread's cumulative tokens), priced at base rates.
    Aggregate,
}

/// Estimate the USD cost of ONE provider call's `usage`.
///
/// The lookup is **source-major**, not key-major: the configured
/// [`CostEstimator`] is offered *every* key the call could be filed under
/// before the static [`model_capabilities`](crate::model_capabilities) table
/// is offered any of them. A key-major walk would let the first key that
/// happens to hit the compiled-in table short-circuit the search, which is
/// exactly the case for the OpenRouter-slug models the static table already
/// carries: their provenance pair resolves statically, and the fresher feed
/// price filed under a derived key would never be reached — a run would keep
/// budgeting at a stale rate long after the real one moved.
///
/// Returns `None` when no source prices the call, so callers treat an
/// un-priced model as "cost unknown" rather than free.
#[must_use]
pub(super) fn estimate_cost_usd(
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    usage: &TokenUsage,
) -> Option<f64> {
    estimate_scoped(pricing, provenance, usage, UsageScope::Call)
}

/// Estimate the USD cost of a SUMMED `usage` — a thread's cumulative tokens.
///
/// See [`UsageScope`]: a sum is not a context size, so this never selects a
/// long-context tier from it.
#[must_use]
pub(super) fn estimate_aggregate_cost_usd(
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    usage: &TokenUsage,
) -> Option<f64> {
    estimate_scoped(pricing, provenance, usage, UsageScope::Aggregate)
}

fn estimate_scoped(
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    usage: &TokenUsage,
    scope: UsageScope,
) -> Option<f64> {
    let usage = Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
    };
    let provider = provenance.provider.as_str();
    let model = provenance.model.as_str();

    if let Some(estimator) = pricing
        && let Some(cost) =
            catalog_candidates(provider, model).find_map(|(provider, model)| match scope {
                UsageScope::Call => estimator.estimate_cost_usd(provider, model, &usage),
                UsageScope::Aggregate => {
                    estimator.estimate_aggregate_cost_usd(provider, model, &usage)
                }
            })
    {
        return Some(cost);
    }

    // The compiled-in table prices every model at one flat rate, so the scope
    // does not change what it answers.
    static_candidates(provider, model).find_map(|(provider, model)| {
        crate::model_capabilities::get_model_capabilities(provider, model)?
            .estimate_cost_usd(&usage)
    })
}

/// The keys a dynamic catalog may file this provenance under: everything the
/// static table is keyed by, plus the keys the third-party feeds derive from a
/// vendor-slug model id.
///
/// The derived keys are offered to the catalog **only**. Probing them against
/// the static table would change what an un-configured loop reports: an
/// OpenRouter-routed `anthropic/claude-fable-5` would start pricing at
/// Anthropic's direct-API rate, which is not what that run pays. Without a
/// catalog the loop must price exactly as it always has.
fn catalog_candidates<'a>(
    provider: &'a str,
    model: &'a str,
) -> impl Iterator<Item = (&'a str, &'a str)> {
    std::iter::once((provider, model))
        .chain(feed_service_keys(provider, model).map(move |service| (service, model)))
        .chain(backend_aliases(provider, model).map(move |backend| (backend, model)))
        .chain(feed_slug_candidates(model))
}

/// The names a feed files a *service* under, when the provider serves models
/// through one rather than calling a vendor directly.
///
/// `VertexProvider` reports `vertex`, but models.dev keys Vertex under the
/// service names Google uses: `google-vertex-anthropic` for the Claude SKUs it
/// resells and `google-vertex` for the rest (both sections carry the
/// `@`-suffixed Vertex model ids, which no direct-vendor section has). Only a
/// raw `google` key is remapped by `map_modelsdev_provider`, so these arrive
/// verbatim.
///
/// They come before the direct-vendor aliases (`anthropic` / `gemini`): a
/// Vertex SKU is billed at Google's rate, so Google's rows answer first, and
/// the vendor's own rows are the approximation behind them.
fn feed_service_keys(provider: &str, model: &str) -> impl Iterator<Item = &'static str> {
    let services: &'static [&'static str] = match provider {
        "vertex" if model.starts_with("claude-") => &["google-vertex-anthropic", "google-vertex"],
        "vertex" => &["google-vertex"],
        _ => &[],
    };
    services.iter().copied()
}

/// The keys the feeds may file a vendor-slug model id (`z-ai/glm-5.1`) under,
/// route first.
///
/// Both feeds file a *routed* row under the route, slug intact —
/// `("openrouter", "moonshotai/kimi-k2.6")` — so that is the first key tried:
/// a run that pays a router's rate must be budgeted at that rate.
///
/// The vendor key (`("moonshotai", "kimi-k2.6")`) is the fallback behind it,
/// and reaches a *native* vendor row: models.dev also publishes the vendor's
/// own section, where the same model sits at the vendor's direct price. That
/// is an approximation of what a routed run pays — close, and far better than
/// no estimate — so it is only consulted once the route key has missed. The
/// `google` remap mirrors `map_modelsdev_provider`, so the halves match the
/// keys that parser emits for a native section.
///
/// The limit of this inference: a slash in a model id is read as a route, but
/// only the provider knows whether it is one. A custom `base_url` pointing at
/// another OpenAI-compatible host that happens to use slash-qualified ids
/// (Groq's `openai/gpt-oss-120b`, say) resolves to the `openrouter` key and is
/// then priced at `OpenRouter`'s rate for that model rather than the host's. The
/// provenance carries nothing that could distinguish the two; only a pricing
/// key supplied by the provider can.
fn feed_slug_candidates(model: &str) -> impl Iterator<Item = (&str, &str)> {
    let route_key = model.contains('/').then_some(("openrouter", model));
    route_key.into_iter().chain(vendor_slug_key(model))
}

/// The provider/model keys the static capability table may file this
/// provenance under, most specific first: the pair the provider reported, then
/// the canonical backend(s) that provider name is known to serve.
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
fn static_candidates<'a>(
    provider: &'a str,
    model: &'a str,
) -> impl Iterator<Item = (&'a str, &'a str)> {
    std::iter::once((provider, model))
        .chain(backend_aliases(provider, model).map(move |backend| (backend, model)))
}

/// The canonical backend(s) a transport-specific provider name serves. See
/// [`static_candidates`] for why each alias exists.
fn backend_aliases(provider: &str, model: &str) -> impl Iterator<Item = &'static str> {
    let backends: &'static [&'static str] = match provider {
        "openai-responses" | "openai-codex" => &["openai"],
        "vertex" if model.starts_with("claude-") => &["anthropic"],
        "vertex" => &["gemini"],
        "cloudflare-ai-gateway" | "record-replay" => &["anthropic", "openai", "gemini"],
        _ => &[],
    };
    backends.iter().copied()
}

/// Split a vendor-slug model id (`z-ai/glm-5.1`) into the vendor key a feed's
/// native section stores it under (`z-ai` / `glm-5.1`).
///
/// Open models (z.ai, Moonshot, `DeepSeek`, `MiniMax`, …) are served through
/// `OpenAIProvider`, so their provenance is `openai` plus the full slug the
/// caller passed, and no feed files them under that pair. The `google` remap
/// mirrors `map_modelsdev_provider`.
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
        state.accumulated_cost_usd =
            estimate_aggregate_cost_usd(pricing, provenance, pre_delta_total);
    }
    if let Some(delta_cost) = estimate_cost_usd(pricing, provenance, delta) {
        state.accumulated_cost_usd = Some(state.accumulated_cost_usd.unwrap_or(0.0) + delta_cost);
    }
}

/// The run's estimated cost so far: the per-call accumulated total when
/// tracked, falling back to repricing the aggregate usage at the current
/// provenance (legacy snapshots / states that never saw a priced call —
/// best-effort, may misprice history across model rotations).
///
/// The fallback prices a SUM, not a call, so it stays on base rates: the
/// accumulated total is the only figure that reflects the tier each individual
/// call actually paid.
#[must_use]
pub(super) fn run_cost_usd(
    accumulated: Option<f64>,
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    usage: &TokenUsage,
) -> Option<f64> {
    accumulated.or_else(|| estimate_aggregate_cost_usd(pricing, provenance, usage))
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
    use crate::types::{AgentState, ThreadId};
    use agent_sdk_providers::model_catalog::{parse_modelsdev, parse_openrouter};
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

    /// A models.dev *native* row for a model the static table also prices
    /// (`openai` / `gpt-4o`), carrying only an input rate.
    const MODELSDEV_PARTIAL_FIXTURE: &str = r#"{
      "openai": {
        "id": "openai",
        "models": {
          "gpt-4o": {
            "id": "gpt-4o",
            "cost": { "input": 1000 }
          }
        }
      }
    }"#;

    /// `OpenRouter`'s listing for models that are also callable directly. The
    /// rates here are the router's, not the vendor's: `gpt-4o` at $15/M output
    /// against the static direct rate of $5/M.
    const OPENROUTER_DIRECT_MODELS_FIXTURE: &str = r#"{
      "data": [
        {
          "id": "openai/gpt-4o",
          "context_length": 128000,
          "pricing": { "prompt": "0.00001", "completion": "0.000015" }
        }
      ]
    }"#;

    /// `z-ai/glm-5.1` is in the bundled static table at $0.98/M input and
    /// $3.08/M output. This feed body reprices it 10× — the shape of a router
    /// raising its rate after the SDK was compiled.
    const OPENROUTER_REPRICED_FIXTURE: &str = r#"{
      "data": [
        {
          "id": "z-ai/glm-5.1",
          "context_length": 202752,
          "pricing": { "prompt": "0.0000098", "completion": "0.0000308" }
        }
      ]
    }"#;

    /// models.dev keeps the outer service key, so an `OpenRouter` route is
    /// stored whole under `openrouter`, while the same model's native vendor
    /// row lives under `moonshotai` at a different (cheaper) price.
    const MODELSDEV_ROUTE_FIXTURE: &str = r#"{
      "openrouter": {
        "id": "openrouter",
        "models": {
          "moonshotai/kimi-k2.6": {
            "id": "moonshotai/kimi-k2.6",
            "cost": { "input": 0.95, "output": 4.0 }
          }
        }
      },
      "moonshotai": {
        "id": "moonshotai",
        "models": {
          "kimi-k2.6": {
            "id": "kimi-k2.6",
            "cost": { "input": 0.66, "output": 3.41 }
          }
        }
      }
    }"#;

    struct FixtureFeed(Vec<CatalogEntry>);

    #[async_trait]
    impl ModelCatalogSource for FixtureFeed {
        async fn fetch(&self) -> Result<Vec<CatalogEntry>> {
            Ok(self.0.clone())
        }
    }

    async fn registry_from_entries(entries: Vec<CatalogEntry>) -> Result<ModelRegistry> {
        let registry = ModelRegistry::new();
        registry.refresh(&FixtureFeed(entries)).await?;
        Ok(registry)
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
        let registry = registry_from_entries(parse_modelsdev(MODELSDEV_PARTIAL_FIXTURE)?).await?;
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

    /// A direct call must never be priced from the router's listing for the
    /// same model. `OpenRouter` rows are keyed by route, so a direct
    /// provenance derives no key that reaches them and falls through to the
    /// static (direct) rate.
    #[tokio::test]
    async fn direct_call_is_not_priced_at_the_routers_rate() -> Result<()> {
        let registry = registry_from(OPENROUTER_DIRECT_MODELS_FIXTURE).await?;
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };

        // Direct: static rate ($1.25/M in + $5/M out = $6.25), never the
        // router's ($10/M + $15/M = $25).
        let direct = AuditProvenance::new("openai", "gpt-4o");
        let cost = estimate_cost_usd(Some(&registry), &direct, &usage)
            .context("gpt-4o is priced in the static table")?;
        assert!((cost - 6.25).abs() < 1e-9, "unexpected direct cost: {cost}");
        let uncatalogued = estimate_cost_usd(None, &direct, &usage)
            .context("gpt-4o is priced in the static table")?;
        assert!((cost - uncatalogued).abs() < 1e-12);

        // The same model called THROUGH the router does pay the router's rate,
        // reached by the route key.
        let routed = AuditProvenance::new("openai", "openai/gpt-4o");
        let routed_cost = estimate_cost_usd(Some(&registry), &routed, &usage)
            .context("the route key must price a routed call")?;
        assert!(
            (routed_cost - 25.0).abs() < 1e-9,
            "unexpected routed cost: {routed_cost}"
        );
        Ok(())
    }

    /// The source-major property: for a slug model the static table already
    /// carries, a repriced feed row still wins. A key-major walk would resolve
    /// the provenance pair statically and never reach the feed's key.
    #[tokio::test]
    async fn feed_price_beats_a_stale_static_price_for_a_known_slug_model() -> Result<()> {
        let registry = registry_from(OPENROUTER_REPRICED_FIXTURE).await?;
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let provenance = AuditProvenance::new("openai", "z-ai/glm-5.1");

        // Compiled-in rate: $0.98/M in + $3.08/M out over 1M each = $4.06.
        let stale = estimate_cost_usd(None, &provenance, &usage)
            .context("the static table prices this slug model")?;
        assert!(
            (stale - 4.06).abs() < 1e-9,
            "unexpected static cost: {stale}"
        );

        // Feed rate, 10× higher: $9.80/M in + $30.80/M out = $40.60.
        let fresh = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the feed prices this slug model")?;
        assert!(
            (fresh - 40.60).abs() < 1e-9,
            "unexpected feed cost: {fresh}"
        );

        // A $5 cap is within the stale rate but well past the real one: the
        // run must stop.
        let limits = UsageLimits {
            max_cost_usd: Some(5.0),
            ..Default::default()
        };
        assert!(status(Some(&limits), None, &provenance, &usage, None).is_none());
        let (limit, _) = status(Some(&limits), Some(&registry), &provenance, &usage, None)
            .context("the fresh feed price must trip the cost limit")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    /// A routed model is budgeted at the route's price, not at the vendor's
    /// native price for the same model.
    #[tokio::test]
    async fn openrouter_routed_model_prices_at_the_route_key() -> Result<()> {
        let registry = registry_from_entries(parse_modelsdev(MODELSDEV_ROUTE_FIXTURE)?).await?;
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let provenance = AuditProvenance::new("openai", "moonshotai/kimi-k2.6");

        // Route row: $0.95/M in + $4.00/M out = $4.95. The vendor's native row
        // ($0.66 + $3.41 = $4.07) is the wrong answer here — the run pays the
        // router.
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the models.dev route key must price this call")?;
        assert!((cost - 4.95).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    /// models.dev keys Vertex under Google's service names. A Vertex-only SKU
    /// (the `@`-suffixed ids no direct-vendor section carries) is priced only
    /// there, so without the service key it could never trip a budget.
    #[tokio::test]
    async fn vertex_sku_prices_from_the_google_vertex_feed_key() -> Result<()> {
        const MODELSDEV_VERTEX_FIXTURE: &str = r#"{
          "google-vertex-anthropic": {
            "id": "google-vertex-anthropic",
            "models": {
              "claude-haiku-4-5@20251001": {
                "id": "claude-haiku-4-5@20251001",
                "cost": { "input": 1, "output": 5 }
              }
            }
          }
        }"#;

        let registry = registry_from_entries(parse_modelsdev(MODELSDEV_VERTEX_FIXTURE)?).await?;
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let provenance = AuditProvenance::new("vertex", "claude-haiku-4-5@20251001");

        // The static table has no Vertex-suffixed SKU, and the `anthropic`
        // alias cannot reach one either.
        assert!(estimate_cost_usd(None, &provenance, &usage).is_none());

        // $1/M in + $5/M out over 1M each = $6.
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the google-vertex-anthropic key must price this SKU")?;
        assert!((cost - 6.0).abs() < 1e-9, "unexpected cost: {cost}");

        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        let (limit, _) = status(Some(&limits), Some(&registry), &provenance, &usage, None)
            .context("a Vertex SKU must be able to trip the cost limit")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    /// A long-context call is budgeted at the tier it falls in. At the base
    /// rate the same call reads as roughly half its true cost — enough to sail
    /// past a cap it has actually exceeded.
    #[tokio::test]
    async fn long_context_call_trips_the_budget_at_its_tier_rate() -> Result<()> {
        const MODELSDEV_TIERED_FIXTURE: &str = r#"{
          "openai": {
            "id": "openai",
            "models": {
              "gpt-5.4": {
                "id": "gpt-5.4",
                "cost": {
                  "input": 2.5,
                  "output": 15,
                  "tiers": [
                    {
                      "input": 5,
                      "output": 22.5,
                      "tier": { "type": "context", "size": 272000 }
                    }
                  ]
                }
              }
            }
          }
        }"#;

        let registry = registry_from_entries(parse_modelsdev(MODELSDEV_TIERED_FIXTURE)?).await?;
        let provenance = AuditProvenance::new("openai", "gpt-5.4");

        // 400K input (past the 272K threshold) + 100K output.
        // Tier: 0.4*5 + 0.1*22.5 = 4.25. Base would say 0.4*2.5 + 0.1*15 = 2.5.
        let usage = TokenUsage {
            input_tokens: 400_000,
            output_tokens: 100_000,
            ..Default::default()
        };
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the tiered feed row must price this call")?;
        assert!((cost - 4.25).abs() < 1e-9, "unexpected cost: {cost}");

        // The loop folds each call through `accumulate_cost`, which prices it
        // as a call — so the accumulator carries the tier rate the call paid.
        let mut state = AgentState::new(ThreadId::new());
        accumulate_cost(
            &mut state,
            Some(&registry),
            &provenance,
            &TokenUsage::default(),
            &usage,
        );
        let accumulated = state
            .accumulated_cost_usd
            .context("the call must accumulate cost")?;
        assert!(
            (accumulated - 4.25).abs() < 1e-9,
            "unexpected accumulated cost: {accumulated}"
        );

        // A $3 cap sits between the tier rate ($4.25) and the base rate
        // ($2.50): only pricing the call at its tier trips it.
        let limits = UsageLimits {
            max_cost_usd: Some(3.0),
            ..Default::default()
        };
        let (limit, _) = status(
            Some(&limits),
            Some(&registry),
            &provenance,
            &usage,
            state.accumulated_cost_usd,
        )
        .context("the tier rate must trip the cost limit")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    /// A long-context call routed through `OpenRouter` is budgeted at the
    /// override rate the route actually charges, reached through the route key.
    #[tokio::test]
    async fn openrouter_long_context_override_trips_the_budget() -> Result<()> {
        const OPENROUTER_OVERRIDE_FIXTURE: &str = r#"{
          "data": [
            {
              "id": "google/gemini-2.5-pro",
              "context_length": 1048576,
              "pricing": {
                "prompt": "0.00000125",
                "completion": "0.00001",
                "overrides": [
                  {
                    "min_prompt_tokens": 200000,
                    "prompt": "0.0000025",
                    "completion": "0.000015"
                  }
                ]
              }
            }
          ]
        }"#;

        let registry = registry_from(OPENROUTER_OVERRIDE_FIXTURE).await?;
        let provenance = AuditProvenance::new("openai", "google/gemini-2.5-pro");

        // 400K prompt (past the 200K override bound) + 100K output.
        // Override: 0.4*2.5 + 0.1*15 = 2.50. Base would say 0.4*1.25 + 0.1*10 = 1.50.
        let usage = TokenUsage {
            input_tokens: 400_000,
            output_tokens: 100_000,
            ..Default::default()
        };
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the override must price this call")?;
        assert!((cost - 2.5).abs() < 1e-9, "unexpected cost: {cost}");

        // The loop folds the call per-call, so the accumulator carries the
        // override rate; a $2 cap sits between it and the base rate.
        let mut state = AgentState::new(ThreadId::new());
        accumulate_cost(
            &mut state,
            Some(&registry),
            &provenance,
            &TokenUsage::default(),
            &usage,
        );
        let limits = UsageLimits {
            max_cost_usd: Some(2.0),
            ..Default::default()
        };
        let (limit, _) = status(
            Some(&limits),
            Some(&registry),
            &provenance,
            &usage,
            state.accumulated_cost_usd,
        )
        .context("the override rate must trip the cost limit")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    /// Repricing a thread's SUMMED usage (the legacy-snapshot seed and the
    /// `run_cost_usd` fallback) must not read the sum as a context size: three
    /// 100K calls sum past a 272K threshold no single call ever reached, and
    /// tier-pricing that sum would invent a long-context bill the thread never
    /// paid — enough to trip a budget it is nowhere near.
    #[tokio::test]
    async fn aggregate_repricing_stays_on_the_base_band() -> Result<()> {
        const MODELSDEV_TIERED_FIXTURE: &str = r#"{
          "openai": {
            "id": "openai",
            "models": {
              "gpt-5.4": {
                "id": "gpt-5.4",
                "cost": {
                  "input": 2.5,
                  "output": 15,
                  "tiers": [
                    {
                      "input": 5,
                      "output": 22.5,
                      "tier": { "type": "context", "size": 272000 }
                    }
                  ]
                }
              }
            }
          }
        }"#;

        let registry = registry_from_entries(parse_modelsdev(MODELSDEV_TIERED_FIXTURE)?).await?;
        let provenance = AuditProvenance::new("openai", "gpt-5.4");

        // A thread's cumulative usage: three 100K-input calls, no output.
        let aggregate = TokenUsage {
            input_tokens: 300_000,
            output_tokens: 0,
            ..Default::default()
        };

        // Base band: 0.3 * $2.5 = $0.75. The tier would say $1.50.
        let cost = run_cost_usd(None, Some(&registry), &provenance, &aggregate)
            .context("the feed prices this model")?;
        assert!(
            (cost - 0.75).abs() < 1e-9,
            "unexpected aggregate cost: {cost}"
        );

        // A $1 cap sits between the two: repricing at the tier would trip it.
        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        assert!(
            status(
                Some(&limits),
                Some(&registry),
                &provenance,
                &aggregate,
                None
            )
            .is_none(),
            "a healthy thread must not be killed by a phantom long-context bill",
        );
        Ok(())
    }

    /// Without a catalog the derived keys must not be probed at all: an
    /// OpenRouter-routed model the static table does not carry stays un-priced
    /// instead of silently billing at the vendor's direct-API rate.
    #[test]
    fn no_catalog_leaves_a_routed_model_unpriced() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let routed = AuditProvenance::new("openai", "anthropic/claude-fable-5");
        assert!(
            estimate_cost_usd(None, &routed, &usage).is_none(),
            "the static table must not answer for a vendor-split key",
        );

        // The direct-API pair for the same model is priced, which is what the
        // derived key would have (wrongly) picked up.
        let direct = AuditProvenance::new("anthropic", "claude-fable-5");
        assert!(estimate_cost_usd(None, &direct, &usage).is_some());
    }
}
