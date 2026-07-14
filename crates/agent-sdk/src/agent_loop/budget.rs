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
use std::borrow::Cow;

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

    if let Some(estimator) = pricing {
        let candidates: Vec<(Cow<'_, str>, Cow<'_, str>)> =
            catalog_candidates(provider, model).collect();
        let override_estimate = |provider: &str, model: &str| match scope {
            UsageScope::Call => estimator.estimate_override_cost_usd(provider, model, &usage),
            UsageScope::Aggregate => {
                estimator.estimate_override_aggregate_cost_usd(provider, model, &usage)
            }
        };
        let feed_estimate = |provider: &str, model: &str| match scope {
            UsageScope::Call => estimator.estimate_feed_cost_usd(provider, model, &usage),
            UsageScope::Aggregate => {
                estimator.estimate_feed_aggregate_cost_usd(provider, model, &usage)
            }
        };

        // Override authority is POSITION-SCOPED: an override has absolute
        // authority only down to its own key's specificity — it never displaces
        // a more-specific live feed price. Candidates are most-specific-first
        // (provenance → gateway → service → aliases → route → vendor-split), so
        // for the first override at index `k`, if no candidate BEFORE `k` has a
        // live feed row the override is the most specific price and wins
        // outright; but if an earlier candidate does, that override belongs to a
        // less-specific fallback context (e.g. a negotiated *direct*-vendor rate
        // reached via a gateway call's vendor-split alias) and must not pre-empt
        // the real price the call actually pays.
        let first_override =
            candidates
                .iter()
                .enumerate()
                .find_map(|(index, (provider, model))| {
                    override_estimate(provider, model)
                        .filter(|cost| is_usable_estimate(*cost))
                        .map(|cost| (index, cost))
                });
        if let Some((index, override_cost)) = first_override {
            let more_specific_feed = candidates[..index].iter().any(|(provider, model)| {
                feed_estimate(provider, model).is_some_and(is_usable_estimate)
            });
            if !more_specific_feed {
                return Some(override_cost);
            }
        }

        // Otherwise the MAX of every FEED candidate estimate (overrides
        // excluded), not the first that resolves. A preferred candidate (route
        // key, gateway key) can carry a flat row where a later candidate — the
        // vendor's own — publishes the tier a long call falls in: models.dev's
        // `openrouter/google/gemini-2.5-pro` route row is flat while its native
        // `gemini-2.5-pro` carries the ≥200K band. First-Some would take the
        // flat route estimate and under-bill the long call. Each estimate is
        // computed WHOLLY from one row, so no rates are blended; taking the
        // larger never bills below the best evidence in the candidate set, and
        // the route/service price still wins whenever it is genuinely the higher
        // one (see [`catalog_candidates`]).
        let best = candidates
            .iter()
            .filter_map(|(provider, model)| feed_estimate(provider, model))
            .filter(|cost| is_usable_estimate(*cost))
            .max_by(f64::total_cmp);
        if let Some(cost) = best {
            return Some(cost);
        }
    }

    // The compiled-in table prices every model at one flat rate, so the scope
    // does not change what it answers.
    static_candidates(provider, model).find_map(|(provider, model)| {
        crate::model_capabilities::get_model_capabilities(provider, model)?
            .estimate_cost_usd(&usage)
    })
}

/// Whether a candidate estimate may be used or entered into the max.
///
/// Rejects NaN and −∞; keeps +∞. `f64::total_cmp` ranks NaN above every finite
/// value, so an unfiltered NaN would win the max and then compare false against
/// `max_cost_usd`, silently disabling the cap while valid candidates existed.
/// −∞ is a nonsensical price that can only win an empty field, and would then
/// poison the accumulator and compare false against the cap — the same
/// fail-open. +∞ is DIFFERENT: a legitimate unbounded over-estimate (a custom
/// estimator, or a rate×tokens overflow), and it must trip any finite cap, so
/// it is kept.
fn is_usable_estimate(cost: f64) -> bool {
    !cost.is_nan() && cost != f64::NEG_INFINITY
}

/// The value a cost is stored as, so a durable accumulator survives a
/// serialize/restore round trip.
///
/// `serde_json` writes a non-finite float as `null`, which on reload looks like
/// "cost unknown" and loses a budget trip. A `+∞` estimate (a legitimate
/// unbounded over-estimate) therefore clamps to `f64::MAX`: finite, survives
/// JSON, still exceeds any real cap so the trip is preserved, and saturates
/// under further addition (`f64::MAX + x` re-clamps back to `f64::MAX`). NaN and
/// −∞ never reach a stored accumulator — [`is_usable_estimate`] drops them
/// before they can be folded — so only the `+∞` case bites.
pub(super) const fn durable_cost(cost: f64) -> f64 {
    cost.min(f64::MAX)
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
///
/// The order still matters even though the caller takes the **max** estimate
/// across these keys rather than the first that resolves (see
/// [`estimate_cost_usd`]): the max is what keeps a route/service row that lacks
/// a tier from shadowing the vendor's tiered row for a long call, while the
/// route/service price still wins whenever it is genuinely the higher one.
///
/// Items are [`Cow`] because most keys borrow the provenance, but a gateway
/// service key owns a freshly prefixed model id (see
/// [`gateway_model_candidates`]).
fn catalog_candidates<'a>(
    provider: &'a str,
    model: &'a str,
) -> impl Iterator<Item = (Cow<'a, str>, Cow<'a, str>)> {
    std::iter::once((Cow::Borrowed(provider), Cow::Borrowed(model)))
        .chain(
            gateway_model_candidates(provider, model)
                .map(|(service, prefixed)| (Cow::Borrowed(service), Cow::Owned(prefixed))),
        )
        .chain(
            feed_service_keys(provider, model)
                .map(move |service| (Cow::Borrowed(service), Cow::Borrowed(model))),
        )
        .chain(
            backend_aliases(provider, model)
                .map(move |backend| (Cow::Borrowed(backend), Cow::Borrowed(model))),
        )
        .chain(feed_slug_candidates(model).map(|(p, m)| (Cow::Borrowed(p), Cow::Borrowed(m))))
}

/// The keys models.dev files a Cloudflare AI Gateway pass-through row under.
///
/// `CloudflareAIGatewayProvider` reports `("cloudflare-ai-gateway", "claude-sonnet-4")`,
/// but models.dev stores the row as `("cloudflare-ai-gateway", "anthropic/claude-sonnet-4")`
/// — the same service provider, the model id prefixed with the backend vendor.
/// So the model half is transformed (not the provider, as for Vertex): each
/// backend the gateway fronts contributes a `backend/model` candidate.
///
/// These come before the direct-vendor aliases: the gateway's own rate is what
/// the caller pays, so its rows answer first, and the wrapped vendor's direct
/// rows are the approximation behind them. The prefixed model half is
/// vendor-exact, so a backend that does not carry the model simply misses —
/// there is no cross-vendor misprice. Only `anthropic` and `openai` have rows
/// in the gateway section today; a Gemini gateway call finds none and falls
/// through to the direct-vendor alias.
fn gateway_model_candidates<'a>(
    provider: &'a str,
    model: &str,
) -> impl Iterator<Item = (&'a str, String)> {
    let backends: &'static [&'static str] = match provider {
        "cloudflare-ai-gateway" => &["anthropic", "openai"],
        _ => &[],
    };
    backends
        .iter()
        .map(move |backend| (provider, format!("{backend}/{model}")))
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
/// native section stores it under (`zai` / `glm-5.1`).
///
/// Open models (z.ai, Moonshot, `DeepSeek`, `MiniMax`, …) are served through
/// `OpenAIProvider`, so their provenance is `openai` plus the full slug the
/// caller passed, and no feed files them under that pair. The vendor half of an
/// `OpenRouter`-style slug is spelled the router's way, which is not always how
/// the vendor's own models.dev section is keyed; [`normalize_slug_vendor`]
/// bridges the two.
fn vendor_slug_key(model: &str) -> Option<(&str, &str)> {
    let (vendor, model_id) = model.split_once('/')?;
    Some((normalize_slug_vendor(vendor), model_id))
}

/// Map an `OpenRouter` route-slug vendor to the provider name its own
/// (non-routed) models.dev section is filed under, when the two differ.
///
/// The pairs are verified against the live models.dev feed: `google`'s section
/// is remapped to `gemini` by `map_modelsdev_provider`, and z.ai / x.ai /
/// Mistral publish native sections spelled without the router's hyphen or `ai`
/// suffix. A vendor already spelled the same in both feeds (`moonshotai`,
/// `deepseek`, `minimax`, …) passes through untouched. Other route vendors
/// whose section carries prefixed or differently-shaped model ids
/// (`meta-llama`, `qwen`) are deliberately left alone: their model halves do
/// not line up with the native section, so a name remap alone would not reach
/// a correct row.
fn normalize_slug_vendor(vendor: &str) -> &str {
    match vendor {
        "google" => "gemini",
        "z-ai" => "zai",
        "x-ai" => "xai",
        "mistralai" => "mistral",
        other => other,
    }
}

/// Whether a usage delta carries no tokens at all.
#[must_use]
pub(super) const fn usage_is_zero(usage: &TokenUsage) -> bool {
    usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_input_tokens == 0
        && usage.cache_creation_input_tokens == 0
}

/// Fold a usage delta's cost into the state's accumulated total, priced at
/// the provenance that served it.
///
/// `delta_scope` is what `delta` describes: a single provider response
/// ([`UsageScope::Call`], tier-aware) or a sum of several
/// ([`UsageScope::Aggregate`], base rates). A compaction that retried a
/// truncated summary bills two calls into one delta, and pricing that as one
/// call could select a long-context tier neither call reached — so it must be
/// folded as an aggregate.
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
    delta_scope: UsageScope,
) {
    if state.accumulated_cost_usd.is_none() && !usage_is_zero(pre_delta_total) {
        state.accumulated_cost_usd =
            estimate_aggregate_cost_usd(pricing, provenance, pre_delta_total).map(durable_cost);
    }
    let delta_cost = match delta_scope {
        UsageScope::Call => estimate_cost_usd(pricing, provenance, delta),
        UsageScope::Aggregate => estimate_aggregate_cost_usd(pricing, provenance, delta),
    };
    if let Some(delta_cost) = delta_cost {
        // Clamp AFTER the addition so the stored accumulator is always finite:
        // `f64::MAX + delta` can round to `+∞`, which serde would then drop.
        state.accumulated_cost_usd = Some(durable_cost(
            state.accumulated_cost_usd.unwrap_or(0.0) + delta_cost,
        ));
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
///
/// The reported figure is clamped to a finite value ([`durable_cost`]): a
/// `+∞` fallback estimate would otherwise serialize to `null` on the terminal
/// state / event, reading as "cost unknown" instead of "over the cap".
#[must_use]
pub(super) fn run_cost_usd(
    accumulated: Option<f64>,
    pricing: Option<&dyn CostEstimator>,
    provenance: &AuditProvenance,
    usage: &TokenUsage,
) -> Option<f64> {
    accumulated
        .or_else(|| estimate_aggregate_cost_usd(pricing, provenance, usage))
        .map(durable_cost)
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

    /// Prices one candidate key `NaN` and another finitely, to prove a
    /// non-finite estimate cannot poison the max.
    struct NanThenFinitePricing;

    impl CostEstimator for NanThenFinitePricing {
        fn estimate_cost_usd(&self, provider: &str, _model: &str, _usage: &Usage) -> Option<f64> {
            match provider {
                // The preferred (route) candidate returns NaN.
                "openrouter" => Some(f64::NAN),
                // The vendor split returns a finite estimate.
                "vendor" => Some(2.5),
                _ => None,
            }
        }
    }

    /// Prices one candidate key `NaN` and another `+∞` — a legitimate unbounded
    /// over-estimate that must survive the filter and trip any finite cap.
    struct NanThenPosInfPricing;

    impl CostEstimator for NanThenPosInfPricing {
        fn estimate_cost_usd(&self, provider: &str, _model: &str, _usage: &Usage) -> Option<f64> {
            match provider {
                "openrouter" => Some(f64::NAN),
                "vendor" => Some(f64::INFINITY),
                _ => None,
            }
        }
    }

    /// Prices its only resolvable candidate `−∞` — a nonsensical price that must
    /// be dropped, leaving the call un-priced rather than poisoned.
    struct NegInfOnlyPricing;

    impl CostEstimator for NegInfOnlyPricing {
        fn estimate_cost_usd(&self, provider: &str, _model: &str, _usage: &Usage) -> Option<f64> {
            (provider == "vendor").then_some(f64::NEG_INFINITY)
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
    fn non_finite_estimate_does_not_poison_the_max() -> anyhow::Result<()> {
        use anyhow::Context;
        // A slug provenance so the candidate set spans several keys: the route
        // key resolves to NaN, the vendor split to a finite 2.5.
        let provenance = AuditProvenance::new("openai", "vendor/model");
        let usage = one_million_each();

        let cost = estimate_cost_usd(Some(&NanThenFinitePricing), &provenance, &usage)
            .context("a finite candidate must price the call")?;
        assert!(cost.is_finite(), "NaN leaked into the estimate: {cost}");
        assert!((cost - 2.5).abs() < 1e-9, "unexpected cost: {cost}");

        // The cap must still fire — an unfiltered NaN would win the max and
        // then compare false against the limit, silently disabling it.
        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        let (limit, _) = status(
            Some(&limits),
            Some(&NanThenFinitePricing),
            &provenance,
            &usage,
            None,
        )
        .context("the finite estimate must trip the cost limit")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    #[test]
    fn positive_infinity_estimate_survives_and_trips_the_cap() -> anyhow::Result<()> {
        use anyhow::Context;
        let provenance = AuditProvenance::new("openai", "vendor/model");
        let usage = one_million_each();

        // NaN is dropped; +∞ survives the filter as the max.
        let cost = estimate_cost_usd(Some(&NanThenPosInfPricing), &provenance, &usage)
            .context("+inf must survive as the estimate")?;
        assert!(
            cost.is_infinite() && cost > 0.0,
            "expected +inf, got {cost}"
        );

        // +∞ trips any finite cap immediately.
        let limits = UsageLimits {
            max_cost_usd: Some(1_000_000.0),
            ..Default::default()
        };
        let (limit, _) = status(
            Some(&limits),
            Some(&NanThenPosInfPricing),
            &provenance,
            &usage,
            None,
        )
        .context("+inf must trip a finite cost cap")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);

        // A +∞ delta folds into the accumulator clamped to `f64::MAX` — finite,
        // so it survives a serde round trip, and still over any real cap.
        let mut state = AgentState::new(ThreadId::new());
        accumulate_cost(
            &mut state,
            Some(&NanThenPosInfPricing),
            &provenance,
            &TokenUsage::default(),
            &usage,
            UsageScope::Call,
        );
        let accumulated = state
            .accumulated_cost_usd
            .context("the +inf delta must accumulate")?;
        assert!(
            accumulated.is_finite(),
            "accumulator not clamped: {accumulated}"
        );
        assert!((accumulated - f64::MAX).abs() < f64::EPSILON);

        // The clamped accumulator survives a serialize/restore (a bare +∞ would
        // become `null`), and keeps tripping on the restored state.
        let json = serde_json::to_string(&state).context("serialize state")?;
        let restored: crate::types::AgentState =
            serde_json::from_str(&json).context("deserialize state")?;
        assert_eq!(restored.accumulated_cost_usd, Some(f64::MAX));
        let (limit, _) = status(
            Some(&limits),
            Some(&NanThenPosInfPricing),
            &provenance,
            &usage,
            restored.accumulated_cost_usd,
        )
        .context("the restored accumulator must keep tripping")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    #[test]
    fn negative_infinity_estimate_is_dropped() {
        let provenance = AuditProvenance::new("openai", "vendor/model");
        let usage = one_million_each();
        // −∞ is the only candidate estimate; filtering it leaves the catalog
        // with nothing, and the static table has no such model — so the call is
        // un-priced rather than billed at a poisoned −∞.
        assert!(estimate_cost_usd(Some(&NegInfOnlyPricing), &provenance, &usage).is_none());
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
            // `moonshotai` is spelled the same in a route slug and in its own
            // models.dev section, so this exercises the plain vendor split; the
            // spelling remap is covered by
            // `catalog_tests::slug_vendor_spelling_reaches_the_native_section`.
            (provider == "moonshotai" && model == "kimi-9-turbo")
                .then(|| f64::from(usage.input_tokens) + f64::from(usage.output_tokens))
        }
    }

    #[test]
    fn estimator_prices_a_vendor_slug_model_under_its_feed_key() -> anyhow::Result<()> {
        use anyhow::Context;
        // Open models route through the OpenAI provider, so the provenance is
        // `openai` plus the full slug, while the feeds key the model by vendor.
        let provenance = AuditProvenance::new("openai", "moonshotai/kimi-9-turbo");
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
            UsageScope::Call,
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

    /// A user override is authoritative: it wins over a HIGHER feed row under a
    /// derived candidate key, rather than being out-maxed. A user who set a
    /// negotiated/free price must not have their cap tripped on feed data they
    /// overrode away.
    #[tokio::test]
    async fn override_authority_beats_a_higher_feed_row() -> Result<()> {
        use crate::model_capabilities::Pricing;

        // Override on the route key: $0.05/M in + $0.05/M out.
        let registry = ModelRegistry::new().with_override(
            "openrouter",
            "vendor/model",
            CatalogEntry {
                provider: "openrouter".to_owned(),
                model_id: "vendor/model".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(Pricing::flat(0.05, 0.05)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            },
        );
        // Feed row on the vendor alias: $2.50/M each — far higher.
        registry
            .refresh(&FixtureFeed(vec![CatalogEntry {
                provider: "vendor".to_owned(),
                model_id: "model".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(Pricing::flat(2.5, 2.5)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            }]))
            .await?;

        let provenance = AuditProvenance::new("openai", "vendor/model");
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };

        // The override wins outright: 1M*0.05 + 1M*0.05 = $0.10, NOT the feed's
        // $5.00 that a plain max over candidates would pick.
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the override must price this call")?;
        assert!((cost - 0.10).abs() < 1e-9, "unexpected cost: {cost}");

        // A $1 cap must NOT trip on the overridden price (it would on the $5
        // feed row).
        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        assert!(
            status(Some(&limits), Some(&registry), &provenance, &usage, None).is_none(),
            "an overridden price must not trip a cap it is under",
        );
        Ok(())
    }

    /// Override authority is position-scoped: a later, less-specific override
    /// must NOT displace an earlier candidate's genuine feed row. A negotiated
    /// *direct*-vendor override is reached, on a gateway call, only through the
    /// less-specific vendor-split alias — but the gateway has its own feed row
    /// at a more specific candidate, and the call actually pays the gateway
    /// rate, so the override must not leak in.
    #[tokio::test]
    async fn a_later_override_does_not_displace_an_earlier_feed_row() -> Result<()> {
        use crate::model_capabilities::Pricing;

        // A cheap/free DIRECT override on the vendor key (reached last, via the
        // vendor-split alias): $0/M.
        let registry = ModelRegistry::new().with_override(
            "anthropic",
            "claude-sonnet-4",
            CatalogEntry {
                provider: "anthropic".to_owned(),
                model_id: "claude-sonnet-4".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(Pricing::flat(0.0, 0.0)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            },
        );
        // A real gateway feed row at the more-specific gateway candidate.
        registry
            .refresh(&FixtureFeed(vec![CatalogEntry {
                provider: "cloudflare-ai-gateway".to_owned(),
                model_id: "anthropic/claude-sonnet-4".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(Pricing::flat(4.0, 20.0)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            }]))
            .await?;

        let provenance = AuditProvenance::new("cloudflare-ai-gateway", "claude-sonnet-4");
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };

        // The gateway feed ($4 + $20 = $24) prices the call — NOT the $0 direct
        // override reached via the vendor-split alias.
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the gateway feed row must price the call")?;
        assert!((cost - 24.0).abs() < 1e-9, "unexpected cost: {cost}");

        // The cap counts the real gateway price; the free override must not
        // disable it.
        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        let (limit, _) = status(Some(&limits), Some(&registry), &provenance, &usage, None)
            .context("the gateway price must trip the cap")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    /// The SAME direct-vendor override, on a DIRECT call, wins — it is the
    /// provenance key itself (the most-specific candidate), so no earlier feed
    /// can out-specific it.
    #[tokio::test]
    async fn a_direct_override_wins_on_a_direct_call() -> Result<()> {
        use crate::model_capabilities::Pricing;

        let registry = ModelRegistry::new().with_override(
            "anthropic",
            "claude-sonnet-4",
            CatalogEntry {
                provider: "anthropic".to_owned(),
                model_id: "claude-sonnet-4".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(Pricing::flat(0.0, 0.0)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            },
        );
        // A feed row exists for the direct key too, at a real price.
        registry
            .refresh(&FixtureFeed(vec![CatalogEntry {
                provider: "anthropic".to_owned(),
                model_id: "claude-sonnet-4".to_owned(),
                context_window: None,
                max_output_tokens: None,
                pricing: Some(Pricing::flat(3.0, 15.0)),
                pricing_tiers: Vec::new(),
                supports_thinking: None,
            }]))
            .await?;

        let provenance = AuditProvenance::new("anthropic", "claude-sonnet-4");
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };

        // The override IS the provenance key — the most specific candidate — so
        // it wins outright: $0, not the $18 feed row.
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the override must price the direct call")?;
        assert!(cost.abs() < 1e-9, "unexpected cost: {cost}");

        // A $1 cap does not trip on the free direct price.
        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        assert!(status(Some(&limits), Some(&registry), &provenance, &usage, None).is_none());
        Ok(())
    }

    /// A preferred candidate's flat row must not shadow a later candidate's
    /// tiered row for a long call. `models.dev` files the `OpenRouter` route row
    /// for `google/gemini-2.5-pro` flat, while the native `gemini-2.5-pro` row
    /// carries the ≥200K band; a routed call reaches both, and the max of the
    /// two complete estimates bills the tier for a long call while a short call
    /// stays on the flat route rate. (No override exists here, so the max
    /// applies — the same-authority path.)
    #[tokio::test]
    async fn tiered_sibling_is_not_shadowed_by_a_flat_route_row() -> Result<()> {
        // The `openrouter` service section files the route flat (no tiers); the
        // `google` native section (remapped to `gemini`) carries the tier.
        const MODELSDEV_ROUTE_AND_NATIVE_FIXTURE: &str = r#"{
          "openrouter": {
            "id": "openrouter",
            "models": {
              "google/gemini-2.5-pro": {
                "id": "google/gemini-2.5-pro",
                "cost": { "input": 1.25, "output": 10 }
              }
            }
          },
          "google": {
            "id": "google",
            "models": {
              "gemini-2.5-pro": {
                "id": "gemini-2.5-pro",
                "cost": {
                  "input": 1.25,
                  "output": 10,
                  "tiers": [
                    {
                      "input": 2.5,
                      "output": 15,
                      "tier": { "type": "context", "size": 200000 }
                    }
                  ]
                }
              }
            }
          }
        }"#;

        let registry =
            registry_from_entries(parse_modelsdev(MODELSDEV_ROUTE_AND_NATIVE_FIXTURE)?).await?;
        // Open models route through OpenAIProvider: provenance is the router
        // slug under `openai`.
        let provenance = AuditProvenance::new("openai", "google/gemini-2.5-pro");

        // Below the 200K band: route flat and native base agree — 100K in +
        // 100K out = 0.1*1.25 + 0.1*10 = 1.125.
        let short = TokenUsage {
            input_tokens: 100_000,
            output_tokens: 100_000,
            ..Default::default()
        };
        let short_cost = estimate_cost_usd(Some(&registry), &provenance, &short)
            .context("the route row prices a short call")?;
        assert!(
            (short_cost - 1.125).abs() < 1e-9,
            "unexpected short cost: {short_cost}"
        );

        // 300K in + 100K out: the native tier ($2.5/$15) says
        // 0.3*2.5 + 0.1*15 = 2.25, the flat route ($1.25/$10) only
        // 0.3*1.25 + 0.1*10 = 1.375. The max bills the tier — the route row
        // must not shadow it.
        let long = TokenUsage {
            input_tokens: 300_000,
            output_tokens: 100_000,
            ..Default::default()
        };
        let long_cost = estimate_cost_usd(Some(&registry), &provenance, &long)
            .context("the tiered native row must price a long call")?;
        assert!(
            (long_cost - 2.25).abs() < 1e-9,
            "unexpected long cost: {long_cost}"
        );
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

    /// A route slug spells its vendor the router's way (`z-ai`), while the
    /// vendor's own models.dev section is keyed differently (`zai`). When the
    /// route row is absent — a trimmed mirror, or a native-only model — the
    /// fallback must still reach the native row through the spelling remap.
    #[tokio::test]
    async fn slug_vendor_spelling_reaches_the_native_section() -> Result<()> {
        // A models.dev NATIVE `zai` section, no `openrouter` route row for it.
        const MODELSDEV_NATIVE_ZAI_FIXTURE: &str = r#"{
          "zai": {
            "id": "zai",
            "models": {
              "glm-5.1": {
                "id": "glm-5.1",
                "cost": { "input": 0.6, "output": 2.2 }
              }
            }
          }
        }"#;

        let registry =
            registry_from_entries(parse_modelsdev(MODELSDEV_NATIVE_ZAI_FIXTURE)?).await?;
        // Open models route through OpenAIProvider, so the provenance is the
        // full router slug under `openai`.
        let provenance = AuditProvenance::new("openai", "z-ai/glm-5.1");
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };

        // The route key `("openrouter", "z-ai/glm-5.1")` misses; the vendor
        // split `("z-ai", …)` would miss too, but the `z-ai` → `zai` remap
        // reaches the native row. $0.6/M in + $2.2/M out = $2.80.
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the zai native section must price this slug model")?;
        assert!((cost - 2.8).abs() < 1e-9, "unexpected cost: {cost}");
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

    /// models.dev files a Cloudflare AI Gateway row under a backend-prefixed
    /// model id (`anthropic/claude-sonnet-4`), while the gateway reports the
    /// bare model id. The gateway rate is reached via the prefixed candidate,
    /// and it wins over the direct-vendor row that the plain alias would find.
    #[tokio::test]
    async fn gateway_row_prices_via_the_prefixed_model_key() -> Result<()> {
        const MODELSDEV_GATEWAY_FIXTURE: &str = r#"{
          "cloudflare-ai-gateway": {
            "id": "cloudflare-ai-gateway",
            "models": {
              "anthropic/claude-sonnet-4": {
                "id": "anthropic/claude-sonnet-4",
                "cost": { "input": 4, "output": 20 }
              }
            }
          },
          "anthropic": {
            "id": "anthropic",
            "models": {
              "claude-sonnet-4": {
                "id": "claude-sonnet-4",
                "cost": { "input": 3, "output": 15 }
              }
            }
          }
        }"#;

        let registry = registry_from_entries(parse_modelsdev(MODELSDEV_GATEWAY_FIXTURE)?).await?;
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let provenance = AuditProvenance::new("cloudflare-ai-gateway", "claude-sonnet-4");

        // The gateway's own rate ($4 + $20 = $24), reached via the
        // `anthropic/claude-sonnet-4` key — not the direct-vendor row ($3 + $15
        // = $18) the plain `anthropic` alias would find.
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the gateway service row must price this call")?;
        assert!((cost - 24.0).abs() < 1e-9, "unexpected cost: {cost}");

        let limits = UsageLimits {
            max_cost_usd: Some(1.0),
            ..Default::default()
        };
        let (limit, _) = status(Some(&limits), Some(&registry), &provenance, &usage, None)
            .context("a gateway row must be able to trip the cost limit")?;
        assert_eq!(limit, BudgetLimitKind::CostUsd);
        Ok(())
    }

    /// When the gateway section has no row for a model, the direct-vendor row
    /// is the documented fallback — not `None`.
    #[tokio::test]
    async fn gateway_falls_back_to_the_direct_vendor_row() -> Result<()> {
        const MODELSDEV_DIRECT_ONLY_FIXTURE: &str = r#"{
          "anthropic": {
            "id": "anthropic",
            "models": {
              "claude-sonnet-4": {
                "id": "claude-sonnet-4",
                "cost": { "input": 3, "output": 15 }
              }
            }
          }
        }"#;

        let registry =
            registry_from_entries(parse_modelsdev(MODELSDEV_DIRECT_ONLY_FIXTURE)?).await?;
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let provenance = AuditProvenance::new("cloudflare-ai-gateway", "claude-sonnet-4");

        // No gateway row; the `anthropic` alias reaches the direct row ($18).
        let cost = estimate_cost_usd(Some(&registry), &provenance, &usage)
            .context("the direct-vendor alias must price this call")?;
        assert!((cost - 18.0).abs() < 1e-9, "unexpected cost: {cost}");
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
            UsageScope::Call,
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
            UsageScope::Call,
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

    /// A compaction that retried a truncated summary bills two calls, summed
    /// into one delta (see `LlmContextCompactor::summarize_with_usage`). Folded
    /// as one call it would select a long-context tier neither call reached;
    /// `accumulate_cost` with `UsageScope::Aggregate` prices it at base rates.
    #[tokio::test]
    async fn compaction_delta_is_priced_at_base_rates() -> Result<()> {
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

        // Two 150K-prompt summarization calls, summed to 300K — over the 272K
        // threshold, though neither call reached it.
        let summed = TokenUsage {
            input_tokens: 300_000,
            output_tokens: 0,
            ..Default::default()
        };
        let mut state = AgentState::new(ThreadId::new());
        accumulate_cost(
            &mut state,
            Some(&registry),
            &provenance,
            &TokenUsage::default(),
            &summed,
            UsageScope::Aggregate,
        );

        // Base: 0.3 * $2.5 = $0.75, not the tier's 0.3 * $5 = $1.50.
        let accumulated = state
            .accumulated_cost_usd
            .context("the compaction spend must accumulate")?;
        assert!(
            (accumulated - 0.75).abs() < 1e-9,
            "compaction priced at the tier rate: {accumulated}"
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
