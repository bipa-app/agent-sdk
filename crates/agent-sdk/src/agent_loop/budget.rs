//! Run-level token / cost budget accounting.
//!
//! The agent loop reserves no budget up front; instead it checks the
//! cumulative usage at every turn-continuation boundary (see
//! [`check_budget`]) and stops the run before starting another turn once a
//! configured [`UsageLimits`] threshold is crossed. Cost is estimated from
//! the run's provider/model pricing metadata
//! ([`agent_sdk_providers`](crate::model_capabilities)); models without
//! pricing simply report `None` and never trip the cost limit.

use crate::types::{BudgetLimitKind, TokenUsage, UsageLimits};
use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::llm::Usage;

/// Estimate the USD cost of `usage` for the run's provider/model.
///
/// Returns `None` when the provider/model pair has no pricing metadata, so
/// callers treat an un-priced model as "cost unknown" rather than free.
#[must_use]
pub(super) fn estimate_cost_usd(provenance: &AuditProvenance, usage: &TokenUsage) -> Option<f64> {
    let caps = lookup_capabilities(&provenance.provider, &provenance.model)?;
    caps.estimate_cost_usd(&Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
    })
}

/// Resolve pricing metadata for a provenance provider/model pair,
/// normalizing known provider aliases before giving up.
///
/// The capability registry keys entries under the canonical provider names
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
fn lookup_capabilities(
    provider: &str,
    model: &str,
) -> Option<&'static crate::model_capabilities::ModelCapabilities> {
    crate::model_capabilities::get_model_capabilities(provider, model).or_else(|| match provider {
        "openai-responses" | "openai-codex" => {
            crate::model_capabilities::get_model_capabilities("openai", model)
        }
        "vertex" if model.starts_with("claude-") => {
            crate::model_capabilities::get_model_capabilities("anthropic", model)
        }
        "vertex" => crate::model_capabilities::get_model_capabilities("gemini", model),
        "cloudflare-ai-gateway" => ["anthropic", "openai", "gemini"]
            .into_iter()
            .find_map(|backend| crate::model_capabilities::get_model_capabilities(backend, model)),
        _ => None,
    })
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
    provenance: &AuditProvenance,
    pre_delta_total: &TokenUsage,
    delta: &TokenUsage,
) {
    if state.accumulated_cost_usd.is_none() && !usage_is_zero(pre_delta_total) {
        state.accumulated_cost_usd = estimate_cost_usd(provenance, pre_delta_total);
    }
    if let Some(delta_cost) = estimate_cost_usd(provenance, delta) {
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
    provenance: &AuditProvenance,
    usage: &TokenUsage,
) -> Option<f64> {
    accumulated.or_else(|| estimate_cost_usd(provenance, usage))
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
    provenance: &AuditProvenance,
    usage: &TokenUsage,
    accumulated_cost_usd: Option<f64>,
) -> Option<(BudgetLimitKind, Option<f64>)> {
    let limits = usage_limits?;
    let cost = run_cost_usd(accumulated_cost_usd, provenance, usage);
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
        let cost = estimate_cost_usd(&provenance, &usage).context("gpt-4o has pricing")?;
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
        let canonical = estimate_cost_usd(&AuditProvenance::new("openai", "gpt-4o"), &usage)
            .context("gpt-4o has pricing")?;
        let aliased =
            estimate_cost_usd(&AuditProvenance::new("openai-responses", "gpt-4o"), &usage)
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
        let canonical = estimate_cost_usd(&AuditProvenance::new("openai", "gpt-4o"), &usage)
            .context("gpt-4o has pricing")?;
        let aliased = estimate_cost_usd(&AuditProvenance::new("openai-codex", "gpt-4o"), &usage)
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
        let canonical =
            estimate_cost_usd(&AuditProvenance::new("anthropic", "claude-fable-5"), &usage)
                .context("claude-fable-5 has pricing")?;
        let aliased = estimate_cost_usd(&AuditProvenance::new("vertex", "claude-fable-5"), &usage)
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
        let canonical =
            estimate_cost_usd(&AuditProvenance::new("gemini", "gemini-3.1-pro"), &usage)
                .context("gemini-3.1-pro has pricing")?;
        let aliased = estimate_cost_usd(&AuditProvenance::new("vertex", "gemini-3.1-pro"), &usage)
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
        let canonical =
            estimate_cost_usd(&AuditProvenance::new("anthropic", "claude-fable-5"), &usage)
                .context("claude-fable-5 has pricing")?;
        let aliased = estimate_cost_usd(
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
        let canonical = estimate_cost_usd(&AuditProvenance::new("openai", "gpt-4o"), &usage)
            .context("gpt-4o has pricing")?;
        let aliased = estimate_cost_usd(
            &AuditProvenance::new("cloudflare-ai-gateway", "gpt-4o"),
            &usage,
        )
        .context("gateway gpt-* must resolve to the openai catalog entry")?;
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
        assert!(estimate_cost_usd(&provenance, &usage).is_none());
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
