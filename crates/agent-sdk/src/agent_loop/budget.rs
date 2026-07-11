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
/// `openai-responses` and `openai-codex` serve `openai` models, and `vertex`
/// serves `anthropic` (`claude-*`) or `gemini` models. Without the alias
/// step those runs would silently report `None` cost and
/// [`crate::types::UsageLimits::max_cost_usd`] could never trip. The alias
/// set mirrors `LlmProvider::capabilities` in `agent-sdk-providers`.
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
        _ => None,
    })
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
