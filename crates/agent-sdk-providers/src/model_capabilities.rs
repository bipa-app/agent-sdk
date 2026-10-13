use agent_sdk_foundation::llm::Usage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceStatus {
    Official,
    Derived,
    Unverified,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PricePoint {
    /// USD per 1M tokens.
    pub usd_per_million_tokens: f64,
}

impl PricePoint {
    #[must_use]
    pub const fn new(usd_per_million_tokens: f64) -> Self {
        Self {
            usd_per_million_tokens,
        }
    }

    #[must_use]
    pub fn estimate_cost_usd(self, tokens: u32) -> f64 {
        (f64::from(tokens) / 1_000_000.0) * self.usd_per_million_tokens
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pricing {
    pub input: Option<PricePoint>,
    pub output: Option<PricePoint>,
    /// Rate for input tokens served from cache (`cache_read`).
    pub cached_input: Option<PricePoint>,
    /// Rate for input tokens written INTO the cache (`cache_write`).
    ///
    /// Providers charge a premium for a cache write — Anthropic's is 1.25× the
    /// ordinary input rate — so a source that publishes one prices those
    /// tokens with it. `None` means the source does not publish one, and
    /// cache-creation tokens then bill at the ordinary input rate, which is a
    /// bounded under-estimate of what the provider actually charges.
    pub cache_write: Option<PricePoint>,
    /// Rate a source publishes for reasoning/thinking tokens, when it differs
    /// from the output rate.
    ///
    /// Reasoning tokens ride *inside* the output-token count — no [`Usage`]
    /// field splits them out — so they cannot be billed separately. Output is
    /// therefore billed at `max(output, reasoning)`: on the many rows where
    /// reasoning is cheaper this is the output rate (unchanged), and on the
    /// rows where reasoning is dearer (e.g. `alibaba/qwen3-32b`: output $2.80/M,
    /// reasoning $8.40/M) it lifts the whole output band to the reasoning rate,
    /// so the estimate is never below what the reasoning tokens actually cost.
    /// The exact split needs a reasoning component on [`Usage`]; see the PR
    /// residual.
    pub reasoning: Option<PricePoint>,
    pub notes: Option<&'static str>,
}

/// The three bands an input-token count splits into for billing, given a
/// [`Usage`]. Cache-creation and cache-read counts are components of
/// `input_tokens` for every provider the SDK supports, so the plain band is
/// what is left after both are taken out.
struct InputBands {
    plain: u32,
    cache_read: u32,
    cache_write: u32,
}

impl InputBands {
    fn split(usage: &Usage) -> Self {
        let cache_read = usage.cached_input_tokens.min(usage.input_tokens);
        let remaining = usage.input_tokens.saturating_sub(cache_read);
        let cache_write = usage.cache_creation_input_tokens.min(remaining);
        Self {
            plain: remaining.saturating_sub(cache_write),
            cache_read,
            cache_write,
        }
    }
}

impl Pricing {
    #[must_use]
    pub const fn flat(input: f64, output: f64) -> Self {
        Self {
            input: Some(PricePoint::new(input)),
            output: Some(PricePoint::new(output)),
            cached_input: None,
            cache_write: None,
            reasoning: None,
            notes: None,
        }
    }

    #[must_use]
    pub const fn flat_with_cached(input: f64, output: f64, cached_input: f64) -> Self {
        Self {
            input: Some(PricePoint::new(input)),
            output: Some(PricePoint::new(output)),
            cached_input: Some(PricePoint::new(cached_input)),
            cache_write: None,
            reasoning: None,
            notes: None,
        }
    }

    /// Builder-style: attach the cache-write (`cache_write`) rate.
    #[must_use]
    pub const fn with_cache_write(mut self, cache_write: f64) -> Self {
        self.cache_write = Some(PricePoint::new(cache_write));
        self
    }

    /// Builder-style: attach the reasoning-token rate.
    #[must_use]
    pub const fn with_reasoning(mut self, reasoning: f64) -> Self {
        self.reasoning = Some(PricePoint::new(reasoning));
        self
    }

    #[must_use]
    pub const fn with_notes(mut self, notes: &'static str) -> Self {
        self.notes = Some(notes);
        self
    }

    /// The rate each input band is billed at, falling back to the ordinary
    /// input rate for a band whose own rate this source does not publish.
    ///
    /// A cache read or a cache write is still an input token: pricing it at
    /// the plain input rate is the provider-agnostic approximation, and it is
    /// what the compiled-in table (which carries no cache rates for most
    /// models) has always done.
    const fn band_rate(&self, band: Option<PricePoint>) -> Option<PricePoint> {
        match band {
            Some(rate) => Some(rate),
            None => self.input,
        }
    }

    #[must_use]
    pub fn estimate_cost_usd(&self, usage: &Usage) -> Option<f64> {
        let bands = InputBands::split(usage);

        let mut input_cost: Option<f64> = None;
        let mut add = |rate: Option<PricePoint>, tokens: u32| {
            if let Some(rate) = rate {
                *input_cost.get_or_insert(0.0) += rate.estimate_cost_usd(tokens);
            }
        };
        add(self.input, bands.plain);
        add(self.band_rate(self.cached_input), bands.cache_read);
        add(self.band_rate(self.cache_write), bands.cache_write);

        // Reasoning tokens ride inside `output_tokens` with no field to split
        // them out, so the whole output band bills at the higher of the output
        // and reasoning rates — see [`Pricing::reasoning`]. On a row with no
        // reasoning rate, or one where reasoning is cheaper, this is just the
        // output rate; where reasoning is dearer it lifts the band so the
        // estimate never falls below the reasoning tokens' true cost.
        let output = output_rate(self.output, self.reasoning)
            .map(|p| p.estimate_cost_usd(usage.output_tokens));
        match (input_cost, output) {
            (Some(input), Some(output)) => Some(input + output),
            (Some(input), None) => Some(input),
            (None, Some(output)) => Some(output),
            (None, None) => None,
        }
    }
}

/// The rate the output band bills at: the more expensive of the output and
/// reasoning rates when both are known, since reasoning tokens are billed
/// inside the output count and must never be under-priced.
const fn output_rate(
    output: Option<PricePoint>,
    reasoning: Option<PricePoint>,
) -> Option<PricePoint> {
    match (output, reasoning) {
        (Some(output), Some(reasoning)) => {
            if reasoning.usd_per_million_tokens > output.usd_per_million_tokens {
                Some(reasoning)
            } else {
                Some(output)
            }
        }
        (rate, None) | (None, rate) => rate,
    }
}

/// What a model does with a request that configures no thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultReasoning {
    /// It answers without reasoning unless asked to.
    Off,
    /// It reasons anyway, because its reasoning is mandatory or on by
    /// default. Those reasoning tokens count against `max_tokens` with no
    /// budget the caller can see.
    On,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ModelCapabilities {
    pub provider: &'static str,
    pub model_id: &'static str,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    /// Output budget for a request whose caller sets none, when it differs
    /// from [`max_output_tokens`](Self::max_output_tokens): a gateway row
    /// whose routes share no single ceiling still needs a sensible default.
    /// See [`default_output_tokens`](Self::default_output_tokens).
    pub default_max_tokens: Option<u32>,
    pub pricing: Option<Pricing>,
    pub supports_thinking: bool,
    pub supports_adaptive_thinking: bool,
    /// The model rejects `ThinkingMode::Enabled { budget_tokens }` (the
    /// API returns 400). Adaptive thinking and provider-default thinking
    /// (a bare effort via `ThinkingConfig::default_with_effort`, no
    /// `thinking` object) both remain valid — the model does NOT require
    /// an adaptive request. Callers holding a budget-mode preference must
    /// switch to one of those shapes before dispatch or the provider
    /// fails the request in `validate_thinking_config`.
    pub rejects_budget_thinking: bool,
    pub default_reasoning: DefaultReasoning,
    pub source_url: &'static str,
    pub source_status: SourceStatus,
    pub notes: Option<&'static str>,
}

impl ModelCapabilities {
    #[must_use]
    pub const fn with_context_window(mut self, context_window: Option<u32>) -> Self {
        self.context_window = context_window;
        self
    }

    #[must_use]
    pub const fn with_default_reasoning(mut self, default_reasoning: DefaultReasoning) -> Self {
        self.default_reasoning = default_reasoning;
        self
    }

    /// Output budget for a request whose caller sets none.
    #[must_use]
    pub const fn default_output_tokens(&self) -> Option<u32> {
        match self.default_max_tokens {
            Some(tokens) => Some(tokens),
            None => self.max_output_tokens,
        }
    }

    #[must_use]
    pub fn estimate_cost_usd(&self, usage: &Usage) -> Option<f64> {
        self.pricing
            .as_ref()
            .and_then(|p| p.estimate_cost_usd(usage))
    }
}

const ANTHROPIC_MODELS_URL: &str =
    "https://docs.anthropic.com/en/docs/about-claude/models/all-models";
const OPENAI_MODELS_URL: &str = "https://developers.openai.com/api/docs/models";
const OPENAI_PRICING_URL: &str = "https://developers.openai.com/api/docs/pricing";
const OPENAI_GPT6_ASTRA_URL: &str = "https://developers.openai.com/api/docs/models/gpt-6-astra";
const OPENAI_GPT6_SOL_URL: &str = "https://developers.openai.com/api/docs/models/gpt-6-sol";
const OPENAI_GPT61_SOL_URL: &str = "https://developers.openai.com/api/docs/models/gpt-6.1-sol";
const OPENAI_GPT6_LUNA_URL: &str = "https://developers.openai.com/api/docs/models/gpt-6-luna";
const OPENAI_GPT56_SOL_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-sol";
const OPENAI_GPT56_TERRA_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-terra";
const OPENAI_GPT56_LUNA_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-luna";
const OPENAI_GPT54_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.4";
const OPENAI_GPT53_CODEX_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.3-codex";
const OPENAI_GPT52_PRO_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.2-pro";
const GOOGLE_MODELS_URL: &str = "https://ai.google.dev/gemini-api/docs/models";
const GOOGLE_PRICING_URL: &str = "https://ai.google.dev/gemini-api/docs/pricing";
const ANTHROPIC_OPUS_45_URL: &str = "https://platform.claude.com/docs/en/models/opus-4-5/overview";
const ANTHROPIC_MYTHOS_51_URL: &str =
    "https://platform.claude.com/docs/en/models/mythos-5-1/overview";
const ANTHROPIC_MYTHOS_5_URL: &str = "https://platform.claude.com/docs/en/models/mythos-5/overview";
const OPENAI_GPT55_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.5";
const GOOGLE_GEMINI35_FLASH_URL: &str =
    "https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash";
const GOOGLE_GEMINI4_ARGON_URL: &str = "https://ai.google.dev/gemini-api/docs/latest-model";
const GOOGLE_CHANGELOG_URL: &str = "https://ai.google.dev/gemini-api/docs/changelog";

// Open-model routes. All reached through OpenAIProvider (provider()=="openai"),
// whether via OpenRouter slugs or the native z.ai / Moonshot / MiniMax base URLs.
const OPENROUTER_GLM51_URL: &str = "https://openrouter.ai/z-ai/glm-5.1";
const ZAI_GLM5_PRICING_URL: &str = "https://docs.z.ai/guides/overview/pricing";
const OPENROUTER_KIMI_K26_URL: &str = "https://openrouter.ai/moonshotai/kimi-k2.6";
const OPENROUTER_KIMI_K25_URL: &str = "https://openrouter.ai/moonshotai/kimi-k2.5";
const KIMI_K25_AA_URL: &str = "https://artificialanalysis.ai/models/kimi-k2-5";
const OPENROUTER_KIMI_K2_THINKING_URL: &str = "https://openrouter.ai/moonshotai/kimi-k2-thinking";
const OPENROUTER_KIMI_K3_URL: &str = "https://openrouter.ai/moonshotai/kimi-k3";
const FIREWORKS_KIMI_K3_URL: &str = "https://fireworks.ai/models/fireworks/kimi-k3";
const OPENROUTER_DEEPSEEK_V4_PRO_URL: &str = "https://openrouter.ai/deepseek/deepseek-v4-pro";
const OPENROUTER_DEEPSEEK_V4_FLASH_URL: &str = "https://openrouter.ai/deepseek/deepseek-v4-flash";
const DEEPSEEK_PRICING_URL: &str = "https://api-docs.deepseek.com/quick_start/pricing";
const MINIMAX_PRICING_URL: &str = "https://platform.minimax.io/docs/guides/pricing-paygo";
const OPENROUTER_MINIMAX_M25_URL: &str = "https://openrouter.ai/minimax/minimax-m2.5";

const MODEL_CAPABILITIES: &[ModelCapabilities] = &[
    // Anthropic
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-fable-5-1",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(10.0, 50.0, 0.25).with_notes("Anthropic Fable 5.1 official pricing: $10 input / $50 output per 1M tokens. Cache reads cost $0.25/M (2.5% of base input, versus 10% on other Claude models). Cache writes cost $12.50/M (5m) or $20/M (1h).")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Fable 5.1 is adaptive-only: adaptive thinking is always on (`thinking.type: \"disabled\"` returns 400) and `ThinkingMode::Enabled { budget_tokens }` is rejected by the Anthropic API. The SDK fails fast in validate_thinking_config. Effort defaults to `high` when unset. Raw chain of thought is never returned — thinking blocks arrive empty (the SDK requests thinking display=omitted). Forced tool use (`tool_choice` `any` or `tool`) returns 400. Safety classifiers may decline a request with stop_reason=refusal on an HTTP 200."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-fable-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(10.0, 50.0).with_notes("Anthropic Fable 5 official pricing: $10 input / $50 output per 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Fable 5 is adaptive-only: adaptive thinking is always on (applies even when `thinking` is unset) and `ThinkingMode::Enabled { budget_tokens }` is rejected by the Anthropic API. The SDK fails fast in validate_thinking_config. Raw chain of thought is never returned — thinking blocks arrive empty (the SDK requests thinking display=omitted). Safety classifiers may decline a request with stop_reason=refusal on an HTTP 200."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-mythos-5-1",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(10.0, 50.0, 0.25).with_cache_write(12.5).with_notes("Anthropic Mythos 5.1 official pricing: $10 input / $50 output per 1M tokens. Cache reads cost $0.25/M; cache writes cost $12.50/M (5m) or $20/M (1h).")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MYTHOS_51_URL,
        source_status: SourceStatus::Official,
        notes: Some("The same model and specifications as Fable 5.1, served only to organizations verified through Anthropic's verification programs. Adaptive thinking is always on; effort defaults to `high`."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-mythos-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(10.0, 50.0, 1.0).with_cache_write(12.5).with_notes("Anthropic Mythos 5 official pricing: $10 input / $50 output per 1M tokens. Cache reads cost $1/M; cache writes cost $12.50/M (5m) or $20/M (1h).")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MYTHOS_5_URL,
        source_status: SourceStatus::Official,
        notes: Some("The same model and specifications as Fable 5, served only to organizations verified through Anthropic's verification programs. Adaptive thinking is always on; effort defaults to `high`."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-5-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(4.0, 20.0, 0.2).with_cache_write(5.0).with_notes("Anthropic Opus 5.5 official pricing: $4 input / $20 output per 1M tokens. Cache reads cost $0.20/M (5% of base input); cache writes cost $5/M (5m) or $8/M (1h).")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Opus 5.5 is adaptive-only: adaptive thinking is always on (`thinking.type: \"disabled\"` returns 400) and `ThinkingMode::Enabled { budget_tokens }` is rejected by the Anthropic API. The SDK fails fast in validate_thinking_config. Effort defaults to `medium` when unset; `max_tokens` caps thinking plus response text, so set it large at high effort."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(5.0, 25.0).with_notes("Anthropic Opus 5 official pricing: $5 input / $25 output per 1M tokens. Fast mode (`speed: \"fast\"`, not implemented by this SDK) is billed at $10/$50. Uses the Opus 4.7 tokenizer, which produces ~30% more tokens than Opus 4.6 and earlier for the same text.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Opus 5 rejects budget thinking — extended thinking (`thinking.type: \"enabled\"`) is not offered, so `ThinkingMode::Enabled { budget_tokens }` returns 400 from the Anthropic API (the SDK fails fast in validate_thinking_config). Adaptive is supported but optional: an effort level can be sent without it via `ThinkingConfig::default_with_effort`. When effort is unset the Claude API defaults to `high`."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-8",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(5.0, 25.0).with_notes("Anthropic Opus 4.8 pricing matches the Opus 4.6 tier ($5/$25 per 1M); verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Opus 4.8 rejects budget thinking — `ThinkingMode::Enabled { budget_tokens }` returns 400 from the Anthropic API (the SDK fails fast in validate_thinking_config). Adaptive is supported but optional: an effort level can be sent without it via `ThinkingConfig::default_with_effort`."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-7",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(5.0, 25.0).with_notes("Anthropic Opus 4.7 pricing matches the Opus 4.6 tier ($5/$25 per 1M); verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Opus 4.7 rejects budget thinking — `ThinkingMode::Enabled { budget_tokens }` returns 400 from the Anthropic API (the SDK fails fast in validate_thinking_config). Adaptive is supported but optional: an effort level can be sent without it via `ThinkingConfig::default_with_effort`."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-6",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(5.0, 25.0).with_notes("Anthropic Opus 4.6 pricing from bundled Claude API guidance; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Current Anthropic docs show this model alongside 200K/128K markers."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-5-20251101",
        context_window: Some(200_000),
        max_output_tokens: Some(64_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(5.0, 25.0, 0.5).with_cache_write(6.25).with_notes("Anthropic Opus 4.5 official pricing: $5 input / $25 output per 1M tokens. Cache reads cost $0.50/M; cache writes cost $6.25/M (5m) or $10/M (1h).")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_OPUS_45_URL,
        source_status: SourceStatus::Official,
        notes: Some("Legacy model, retiring no sooner than 2026-11-24. Extended (budget) thinking, not adaptive; the effort parameter defaults to `high`. The alias `claude-opus-4-5` and the Google Cloud ID `claude-opus-4-5@20251101` name this snapshot."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-5-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(2.0, 10.0, 0.1).with_cache_write(2.5).with_notes("Anthropic Sonnet 5.5 official pricing: $2 input / $10 output per 1M tokens. Cache reads cost $0.10/M (5% of base input); cache writes cost $2.50/M (5m) or $4/M (1h).")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Thinking is on by default and `thinking.type: \"disabled\"` returns 400 (Anthropic thinking docs); OpenRouter marks its reasoning mandatory. Default effort `high`. `ThinkingMode::Enabled { budget_tokens }` is rejected by the Anthropic API; the SDK fails fast in validate_thinking_config."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet 5 standard pricing $3/$15 per 1M; introductory $2/$10 through 2026-08-31. A new tokenizer produces ~30% more tokens than Sonnet 4.6, so equivalent-text cost differs even at unchanged per-token rates.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Sonnet 5 is adaptive-only: adaptive thinking is on by default (applies even when `thinking` is unset) and `ThinkingMode::Enabled { budget_tokens }` returns 400 from the Anthropic API — same as Opus 4.8. Non-default sampling params (temperature/top_p/top_k) also return 400 (constraint inherited from Opus 4.7). Uses a new tokenizer (~30% more tokens than Sonnet 4.6)."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-4-6",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Anthropic docs list Sonnet 4.6; user confirmed adaptive thinking support."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-4-5-20250929",
        context_window: Some(200_000),
        max_output_tokens: Some(64_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-haiku-5-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.1, 0.5, 0.01).with_cache_write(0.125).with_notes("Anthropic Haiku 5.5 official pricing for prompts up to 100K tokens: $0.10 input / $0.50 output per 1M tokens, cache reads $0.01/M (10% of input), cache writes $0.125/M (5m) or $0.20/M (1h). Prompts above 100K tokens bill the whole request at $0.50 input / $2.50 output, cache reads $0.05/M, cache writes $0.625/M (5m) or $1/M (1h); this table carries only the base tier.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: true,
        default_reasoning: DefaultReasoning::On,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Haiku 5.5 is adaptive-only: `ThinkingMode::Enabled { budget_tokens }` is rejected by the Anthropic API and the SDK fails fast in validate_thinking_config. Effort defaults to `medium` when unset; `xhigh` and `max` are accepted. Claude Code's model catalog marks it as rejecting `thinking.type: \"disabled\"`, which the SDK never sends. Thinking is on by default (Anthropic thinking docs: \"Claude Haiku 5.5 also has thinking on by default\"); it accepts `thinking.type: \"disabled\"` at effort `high` or below."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-haiku-4-5-20251001",
        context_window: Some(200_000),
        max_output_tokens: Some(64_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(1.0, 5.0).with_notes("Anthropic Haiku tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-4-20250514",
        context_window: Some(200_000),
        max_output_tokens: Some(64_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-20250514",
        context_window: Some(200_000),
        max_output_tokens: Some(32_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(15.0, 75.0).with_notes("Anthropic Opus tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-3-5-sonnet-20241022",
        context_window: Some(200_000),
        max_output_tokens: Some(8_192),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-3-5-haiku-20241022",
        context_window: Some(200_000),
        max_output_tokens: Some(8_192),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(1.0, 5.0).with_notes("Anthropic Haiku tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    // OpenAI
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-6-astra",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(10.0, 50.0, 1.0).with_notes(
            "Standard tier base rates. Cache writes cost $12.50/M input tokens. Requests with more than 272K input tokens cost 2x input and cache rates and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT6_ASTRA_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output. Rejects `none` reasoning effort. Chat Completions does not support function calling; tool calls need the Responses API."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-6-sol",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(2.0, 10.0, 0.2).with_notes(
            "Standard tier base rates. Cache writes cost $2.50/M input tokens. Requests with more than 272K input tokens cost 2x input and cache rates and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT6_SOL_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output. Chat Completions supports function calling only with `reasoning_effort: none`; tools with reasoning need the Responses API."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-6.1-sol",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(2.0, 10.0, 0.1).with_cache_write(2.5).with_notes(
            "Standard tier base rates. Cached input costs $0.10/M (5% of base input, versus 10% on other GPT-6 models). Cache writes cost $2.50/M input tokens. Requests with more than 272K input tokens cost 2x input and cache rates and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT61_SOL_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output. Rejects `none` reasoning effort. Chat Completions does not support function calling; tool calls need the Responses API."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-6-luna",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.1, 0.5, 0.01).with_notes(
            "Standard tier base rates. Cache writes cost $0.125/M input tokens. Requests with more than 272K input tokens cost 2x input and cache rates and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT6_LUNA_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output. Chat Completions supports function calling only with `reasoning_effort: none`; tools with reasoning need the Responses API."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.6",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(5.0, 30.0, 0.5).with_notes(
            "Standard tier base rates. Cache writes cost $6.25/M input tokens. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT56_SOL_URL,
        source_status: SourceStatus::Official,
        notes: Some("Official alias for GPT-5.6 Sol."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.6-sol",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(5.0, 30.0, 0.5).with_notes(
            "Standard tier base rates. Cache writes cost $6.25/M input tokens. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT56_SOL_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.6-terra",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(2.5, 15.0, 0.25).with_notes(
            "Standard tier base rates. Cache writes cost $3.125/M input tokens. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT56_TERRA_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.6-luna",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(1.0, 6.0, 0.1).with_notes(
            "Standard tier base rates. Cache writes cost $1.25/M input tokens. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT56_LUNA_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.5",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(5.0, 30.0, 0.5).with_notes(
            "Standard tier base rates. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT55_URL,
        source_status: SourceStatus::Official,
        notes: Some("Snapshot `gpt-5.5-2026-04-23`. reasoning.effort accepts none, low, medium, high and xhigh; it defaults to medium, so the model reasons unless told not to."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.4",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(2.50, 15.0, 0.25)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENAI_GPT54_URL,
        source_status: SourceStatus::Official,
        notes: Some("OpenAI model docs list 1.05M context, 128K max output, and reasoning.effort support."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.3-codex",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(1.50, 6.0, 0.375)),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT53_CODEX_URL,
        source_status: SourceStatus::Official,
        notes: Some("OpenAI model docs list Responses-only access, a 272K maximum input, 128K maximum output, and reasoning.effort levels."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(1.25, 10.0, 0.125)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5-mini",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.125, 1.0, 0.0125)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5-nano",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.025, 0.20, 0.0025)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.2-instant",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: None,
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENAI_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model exists in OpenAI docs, but pricing was not extracted from the official pricing page in this pass."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.2-thinking",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENAI_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model exists in OpenAI docs, but pricing was not extracted from the official pricing page in this pass."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.2-pro",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(21.0, 168.0)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_GPT52_PRO_URL,
        source_status: SourceStatus::Official,
        notes: Some("Responses-only pro model. Supports medium, high, and xhigh reasoning effort."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.2-codex",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        default_max_tokens: None,
        pricing: None,
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model presence confirmed from OpenAI docs; pricing not yet extracted in this pass."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o3",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(1.0, 4.0)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o3-mini",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.55, 2.20)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o4-mini",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.55, 2.20)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o1",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(7.50, 30.0)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o1-mini",
        context_window: Some(128_000),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.55, 2.20)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context window 128,000 and max output 65,536 from the OpenAI model page (developers.openai.com/api/docs/models/o1-mini)."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4.1",
        context_window: Some(1_000_000),
        max_output_tokens: Some(32_768),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(1.0, 4.0)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context window from model family docs/notes. Max output 32,768 from the OpenAI model page (developers.openai.com/api/docs/models)."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4.1-mini",
        context_window: Some(1_000_000),
        max_output_tokens: Some(32_768),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.20, 0.80)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context window from model family docs/notes. Max output 32,768 from the OpenAI model page (developers.openai.com/api/docs/models)."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4.1-nano",
        context_window: Some(1_000_000),
        max_output_tokens: Some(32_768),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.05, 0.20)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context window from model family docs/notes. Max output 32,768 from the OpenAI model page (developers.openai.com/api/docs/models)."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4o",
        context_window: Some(128_000),
        max_output_tokens: Some(16_384),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(1.25, 5.0)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output from existing runtime assumptions."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4o-mini",
        context_window: Some(128_000),
        max_output_tokens: Some(16_384),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.075, 0.30)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output from existing runtime assumptions."),
    },
    // Gemini
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-4-argon",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(2.0, 10.0, 0.10).with_notes("Introductory rate through 2026-11-30: $2 input / $10 output including thinking / $0.10 cached input per 1M tokens. From 2026-12-01 the rate is $4 / $20 / $0.20. Implicit caching only, on prompts of at least 4,096 tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_GEMINI4_ARGON_URL,
        source_status: SourceStatus::Official,
        notes: Some("GA 2026-10-13. Input, thinking and output share one 1,048,576-token window. A synchronous response defaults to and stops at 65,536 output tokens; longer decodes need background execution on the Interactions API. thinking_level low / medium / high, default medium; `minimal` is not accepted. temperature, top_p and top_k are ignored. Explicit caching (`cachedContents`) is not supported. Not on the Free tier."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.8-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.75, 3.75, 0.075).with_notes("Introductory rate through 2026-12-31: $0.75 input / $3.75 output including thinking / $0.075 cached input per 1M tokens. From 2027-01-01 the rate is $1.50 / $7.50 / $0.15.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("GA 2026-09-02. thinking_level low / medium / high, default medium; `minimal` returns an error (ai.google.dev/gemini-api/docs/models/gemini-3.8-flash). max_output_tokens covers thinking plus output."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.7-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_CHANGELOG_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Deprecated 2026-10-08: Google routes `gemini-3.7-flash` requests to `gemini-3.8-flash`, and the pricing page no longer lists a 3.7 Flash rate. Limits are those of gemini-3.8-flash, the model that serves it."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.6-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.75, 3.75, 0.075).with_notes("Introductory rate through 2026-12-31: $0.75 input / $3.75 output including thinking / $0.075 cached input per 1M tokens. From 2027-01-01 the rate is $1.50 / $7.50 / $0.15.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("thinking_level minimal / low / medium / high, default medium (ai.google.dev/gemini-api/docs/thinking). max_output_tokens covers thinking plus output."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.5-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_GEMINI35_FLASH_URL,
        source_status: SourceStatus::Official,
        notes: Some("Deprecated: Google routes `gemini-3.5-flash` requests to `gemini-3.6-flash`, and the pricing page no longer lists a 3.5 Flash rate. Limits are the model page's 1,048,576 input and 65,536 output tokens."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.5-flash-lite",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.30, 2.50, 0.03).with_notes("Official text / image / video / audio pricing per 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("thinking_level minimal / low / medium / high, default minimal (ai.google.dev/gemini-api/docs/thinking). max_output_tokens covers thinking plus output."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.1-flash-lite",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.25, 1.50, 0.025).with_notes("Official text / image / video pricing per 1M tokens. Audio input is priced separately at $0.50 (cached $0.05).")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Thinks by default: Google states \"Gemini models engage in dynamic thinking by default\" (ai.google.dev/gemini-api/docs/thinking). max_output_tokens covers thinking plus output."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.1-pro-preview",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(2.0, 12.0).with_notes("Official pricing for prompts <= 200K tokens. For prompts > 200K, pricing increases to $4 input / $18 output per 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing sourced from Gemini 3.1 Pro Preview docs. Thinks by default and max_output_tokens covers thinking plus output (ai.google.dev/gemini-api/docs/thinking: \"gemini-3.1-pro-preview On (high)\"); OpenRouter marks its reasoning mandatory."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.1-pro",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(2.0, 12.0).with_notes("Legacy alias retained for compatibility. For prompts > 200K, pricing increases to $4 input / $18 output per 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Legacy Gemini 3.1 Pro alias retained for compatibility; prefer gemini-3.1-pro-preview. Thinks by default and max_output_tokens covers thinking plus output (ai.google.dev/gemini-api/docs/thinking: \"gemini-3.1-pro-preview On (high)\", the model this alias names)."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3-flash-preview",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model presence confirmed from Google docs, but pricing was not extracted in this pass. Thinks by default and max_output_tokens covers thinking plus output (ai.google.dev/gemini-api/docs/thinking: \"gemini-3-flash-preview On (high)\")."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.0-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Legacy Gemini 3.0 Flash model retained for compatibility; prefer gemini-3-flash-preview. Thinks by default and max_output_tokens covers thinking plus output (ai.google.dev/gemini-api/docs/thinking: \"gemini-3-flash-preview On (high)\", the model this alias names)."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-2.5-flash",
        context_window: Some(1_000_000),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.30, 2.50).with_notes("Official text/image/video pricing. Audio input is priced separately at $1.00 / 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Official docs state output pricing includes thinking tokens. Thinks by default and max_output_tokens covers thinking plus output (ai.google.dev/gemini-api/docs/thinking: \"gemini-2.5-flash On\")."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-2.5-pro",
        context_window: Some(1_000_000),
        max_output_tokens: Some(65_536),
        default_max_tokens: None,
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: GOOGLE_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model presence confirmed from Google docs, but pricing was not extracted in this pass. Thinks by default and max_output_tokens covers thinking plus output (ai.google.dev/gemini-api/docs/thinking: \"gemini-2.5-pro On\"); OpenRouter marks its reasoning mandatory."),
    },
    // Open models (z.ai / Moonshot / DeepSeek / MiniMax). All routed through
    // OpenAIProvider, so provider == "openai" and the model_id is the exact
    // string the caller passes (OpenRouter slug or native model id).
    ModelCapabilities {
        provider: "openai",
        model_id: "z-ai/glm-5.1",
        context_window: Some(202_752),
        max_output_tokens: Some(131_072),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.98, 3.08).with_notes("OpenRouter rate for z-ai/glm-5.1: input $0.98/M, output $3.08/M.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENROUTER_GLM51_URL,
        source_status: SourceStatus::Derived,
        notes: Some("GLM-5.1 (z.ai/Zhipu) via OpenRouter slug. Reasoning/thinking model; context 203K (=202,752). max_output 128K from z.ai GLM-5.1 docs, sized generously for hidden reasoning + answer. Released ~Apr 7, 2026."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "glm-5",
        context_window: Some(200_000),
        max_output_tokens: Some(131_072),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(1.0, 3.2).with_notes("Native z.ai pricing: input $1.0/M, output $3.2/M (higher than the OpenRouter GLM-5 rate of $0.60/$1.92).")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: ZAI_GLM5_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Native z.ai constructor model string `glm-5`. Reasoning/thinking model; 200K context, 128K (131072) max output per docs.z.ai/guides/llm/glm-5. Native pricing used for the native route. Released ~Feb 11, 2026."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "moonshotai/kimi-k2.6",
        context_window: Some(262_144),
        max_output_tokens: None,
        default_max_tokens: Some(65_536),
        pricing: Some(Pricing::flat(0.684, 3.42).with_notes("OpenRouter rate for moonshotai/kimi-k2.6: input $0.684/M, output $3.42/M.")),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENROUTER_KIMI_K26_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Exact OpenRouter slug (note the dot). Hybrid model marketed/used as a non-reasoning coding+multimodal model, so supports_thinking=false (use moonshotai/kimi-k2-thinking for the dedicated reasoning model). Context 262,144. OpenRouter endpoints cap completions between 16,384 and 235,929 tokens (Baseten 262,000) and OpenRouter routes a request only to endpoints that support its max_tokens, so max_output_tokens stays None; 65,536 is the default budget for callers that set none. OpenRouter reports reasoning as enabled by default."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "moonshotai/kimi-k2.5",
        context_window: Some(262_144),
        max_output_tokens: None,
        default_max_tokens: Some(32_768),
        pricing: Some(Pricing::flat(0.4, 1.9).with_notes("OpenRouter rate for moonshotai/kimi-k2.5: input $0.40/M, output $1.90/M.")),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENROUTER_KIMI_K25_URL,
        source_status: SourceStatus::Derived,
        notes: Some("OpenRouter route for the model the native constructor names 'kimi-k2.5'. Treated as non-reasoning (visual-coding + agentic tool-calling) on OpenRouter. Context 262,144. OpenRouter endpoints cap completions between 65,536 and 235,929 tokens (Baseten 262,000) and OpenRouter routes a request only to endpoints that support its max_tokens, so max_output_tokens stays None; 32,768 is the default budget for callers that set none. OpenRouter reports reasoning as enabled by default."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "kimi-k2.5",
        context_window: Some(262_144),
        max_output_tokens: Some(32_768),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.6, 3.0).with_notes("Native Moonshot estimate from Artificial Analysis (~$0.58 in / $3.00 out); input rounded up to $0.60 to stay conservative for budget reservation.")),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: KIMI_K25_AA_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Exact native model_id used by the native constructor (Moonshot platform.kimi.ai base_url). Native pricing not on the first-party table (only k2.6 is enumerated); figures derived from Artificial Analysis. Context 262,144; 32768 is a generous within-window completion budget. Reasoning is treated as on by default to match the OpenRouter row for the same weights, which OpenRouter reports as enabled by default; Moonshot's current docs no longer list K2.5, so the native default is unverified."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "kimi-k2-thinking",
        context_window: Some(262_144),
        max_output_tokens: Some(131_072),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.6, 2.5).with_notes("Cross-provider median for kimi-k2-thinking (OpenRouter/Artificial Analysis): input $0.60/M, output $2.50/M, used as a conservative native estimate.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENROUTER_KIMI_K2_THINKING_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Exact native model_id used by the native constructor; a REASONING model (emits hidden chain-of-thought before the answer). Native Moonshot base_url. First-party pricing could not be isolated; figures are the cross-provider median. Context 262,144; max_output 131072 sized generously for reasoning tokens, within the window."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "moonshotai/kimi-k3",
        context_window: Some(1_048_576),
        max_output_tokens: None,
        default_max_tokens: None,
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENROUTER_KIMI_K3_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Exact OpenRouter slug; Baseten serves the same weights as moonshotai/Kimi-K3, which the case-insensitive lookup folds here. Reasoning model with a 1,048,576-token window (OpenRouter model metadata; Baseten documents a 1M window). OpenRouter endpoints cap completions between 65,535 and 943,718 tokens (Baseten 262,144), and OpenRouter routes a request only to endpoints that support its max_tokens, so no single ceiling applies and max_output_tokens stays None: a fixed cap would cut requests some route serves. Callers size output through their own max_tokens. Pricing is None because the static `openai` key cannot tell the gateway base URLs apart and their rates differ; dynamic route catalogs remain the pricing authority. Reasons by default: Moonshot documents reasoning_effort defaulting to `max`, and OpenRouter reports reasoning enabled by default."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "accounts/fireworks/models/kimi-k3",
        context_window: Some(1_048_576),
        max_output_tokens: None,
        default_max_tokens: None,
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: FIREWORKS_KIMI_K3_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Fireworks model path for Kimi K3. Fireworks documents a 1,048,576-token window and no fixed output cap (max_tokens is lowered to fit the window). Thinking metadata mirrors the moonshotai/kimi-k3 row; pricing is None for the same reason. Same weights as moonshotai/kimi-k3, which reasons by default (Moonshot: reasoning_effort defaults to `max`)."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "deepseek/deepseek-v4-pro",
        context_window: Some(1_048_576),
        max_output_tokens: Some(384_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.44, 0.87).with_notes("OpenRouter effective post-promo rate ($0.435 in rounded up to $0.44 / $0.87 out). Pre-promo regular rate was $1.74/$3.48.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENROUTER_DEEPSEEK_V4_PRO_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Primary model named in forge config; exact OpenRouter slug. Large MoE (1.6T total / 49B active), released 2026-04-24. Reasoning/thinking model; DeepSeek returns the answer in `content` and chain-of-thought in a separate `reasoning_content` field, which must be echoed back in subsequent thinking-mode turns or the API returns 400. Max output 384K (DeepSeek ceiling), sized generously for reasoning."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "deepseek-v4-pro",
        context_window: Some(1_048_576),
        max_output_tokens: Some(384_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.44, 0.87, 0.003_625).with_notes("Official DeepSeek pricing: input cache-MISS $0.435/M (rounded up to $0.44), cache-HIT $0.003625/M, output $0.87/M.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: DEEPSEEK_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Native DeepSeek API model id 'deepseek-v4-pro' (no vendor prefix). 1M context, 384K max output. Reasoning/thinking model; separate `reasoning_content` that must be echoed back in multi-turn thinking-mode requests or you get a 400. Legacy ids deepseek-reasoner/deepseek-chat now map to V4-FLASH, not Pro."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "deepseek/deepseek-v4-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(384_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.15, 0.28).with_notes("DeepSeek list rate rounded up ($0.14 in -> $0.15 / $0.28 out) used instead of OpenRouter's lower fluctuating effective rate so consumers never under-reserve budget.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: OPENROUTER_DEEPSEEK_V4_FLASH_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Sibling V4 model (cheaper routing target). Efficiency MoE (284B total / 13B active), released 2026-04-24. Reasoning/thinking model with the same reasoning_content split + mandatory pass-back-or-400 behavior as V4 Pro. Max output 384K per DeepSeek docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "deepseek-v4-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(384_000),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.14, 0.28, 0.002_8).with_notes("Official DeepSeek pricing: input cache-MISS $0.14/M, cache-HIT $0.0028/M, output $0.28/M.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::Off,
        source_url: DEEPSEEK_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Native DeepSeek API model id 'deepseek-v4-flash'. 1M context, 384K max output. Reasoning/thinking model; same content/reasoning_content split and mandatory pass-back in thinking mode. Legacy aliases deepseek-chat/deepseek-reasoner now resolve to this Flash model."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "MiniMax-M2.5",
        context_window: Some(204_800),
        max_output_tokens: Some(131_072),
        default_max_tokens: None,
        pricing: Some(Pricing::flat_with_cached(0.3, 1.2, 0.03).with_notes("Native MiniMax first-party pricing: input $0.30/M, output $1.20/M, cache-read input $0.03/M (platform.minimax.io PAYG).")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: MINIMAX_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Native agent-sdk constructor model string 'MiniMax-M2.5' (api.minimax.io, OpenAI-compatible). Reasoning/thinking model; emits chain-of-thought in <think>...</think> tags and supports interleaved thinking. Context 204,800; max_output 131072 sized generously for hidden reasoning + answer within the window."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "minimax/minimax-m2.5",
        context_window: Some(204_800),
        max_output_tokens: Some(131_072),
        default_max_tokens: None,
        pricing: Some(Pricing::flat(0.15, 1.15).with_notes("OpenRouter rate for minimax/minimax-m2.5: input $0.15/M, output $1.15/M (lower than MiniMax's $0.30/$1.20 first-party rate; OpenRouter prices can fluctuate, so reserve conservatively).")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        rejects_budget_thinking: false,
        default_reasoning: DefaultReasoning::On,
        source_url: OPENROUTER_MINIMAX_M25_URL,
        source_status: SourceStatus::Derived,
        notes: Some("OpenRouter slug 'minimax/minimax-m2.5' (same M2.5 weights as native). Reasoning/thinking model. Context 204,800; max_output 131072 sized generously for hidden reasoning tokens before the answer."),
    },
];

/// Capabilities of `model_id` served by `provider`.
///
/// An exact id wins. Otherwise the id resolves to the row of its snapshot
/// family (the id without a `-YYYYMMDD` or `-YYYY-MM-DD` date, a Google
/// Cloud `@` version or a `-latest` suffix) when that row describes the same
/// snapshot: an undated alias or `-latest` names the row's snapshot, a
/// dated id resolves to a row of the same date, and to an undated row only
/// when the registry lists that date as the snapshot the row describes. Any other snapshot of
/// a registered family may differ from its row, so it stays unknown.
#[must_use]
pub fn get_model_capabilities(
    provider: &str,
    model_id: &str,
) -> Option<&'static ModelCapabilities> {
    let rows = || {
        MODEL_CAPABILITIES
            .iter()
            .filter(move |caps| caps.provider.eq_ignore_ascii_case(provider))
    };
    rows()
        .find(|caps| caps.model_id.eq_ignore_ascii_case(model_id))
        .or_else(|| rows().find(|caps| snapshot_resolves_to(model_id, caps.model_id)))
}

/// Dated snapshots an undated row describes, as `(row, YYYY-MM-DD)`: the
/// snapshot its alias points to on the vendor's model page. Other dated
/// snapshots of that model may have other limits or prices
/// (`gpt-4o-2024-05-13` allows 4,096 output tokens, not 16,384).
const SNAPSHOTS_OF_UNDATED_ROWS: &[(&str, &str)] =
    &[("gpt-5.5", "2026-04-23"), ("gpt-4o", "2024-08-06")];

/// The qualifier a model id carries after its family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SnapshotQualifier {
    /// No date: the family name or a `-latest` alias.
    Undated,
    /// A snapshot date, as `YYYYMMDD`.
    Dated(u32),
    /// A Google Cloud `@` version that is not a date.
    Version,
}

/// Whether `query`, which is not `row_id` itself, names the snapshot the
/// row `row_id` describes.
pub(crate) fn snapshot_resolves_to(query: &str, row_id: &str) -> bool {
    let (query_family, query_snapshot) = split_snapshot(query);
    let (row_family, row_snapshot) = split_snapshot(row_id);
    if !query_family.eq_ignore_ascii_case(row_family) {
        return false;
    }
    match (query_snapshot, row_snapshot) {
        (SnapshotQualifier::Undated, SnapshotQualifier::Undated | SnapshotQualifier::Dated(_)) => {
            true
        }
        (SnapshotQualifier::Dated(query_date), SnapshotQualifier::Dated(row_date)) => {
            query_date == row_date
        }
        (SnapshotQualifier::Dated(query_date), SnapshotQualifier::Undated) => {
            SNAPSHOTS_OF_UNDATED_ROWS.iter().any(|(row, date)| {
                row.eq_ignore_ascii_case(row_id) && dashed_date(date) == Some(query_date)
            })
        }
        (SnapshotQualifier::Version, _) | (_, SnapshotQualifier::Version) => false,
    }
}

/// `model_id` split into its family and its snapshot qualifier: a Google
/// Cloud `@version`, a `-latest` alias suffix, or a trailing snapshot date
/// (`-YYYYMMDD`, or `-YYYY-MM-DD` as `OpenAI` dates them).
fn split_snapshot(model_id: &str) -> (&str, SnapshotQualifier) {
    if let Some((family, version)) = model_id.split_once('@') {
        let qualifier = compact_date(version).map_or(SnapshotQualifier::Version, |date| {
            SnapshotQualifier::Dated(date)
        });
        return (family, qualifier);
    }
    if let Some(family) = model_id.strip_suffix("-latest") {
        return (family, SnapshotQualifier::Undated);
    }
    if let Some((family, date)) = model_id.rsplit_once('-')
        && let Some(date) = compact_date(date)
    {
        return (family, SnapshotQualifier::Dated(date));
    }
    if let Some(family) = model_id.get(..model_id.len().saturating_sub(11))
        && let Some(date) = model_id.get(family.len()..)
        && let Some(date) = date.strip_prefix('-')
        && let Some(date) = dashed_date(date)
    {
        return (family, SnapshotQualifier::Dated(date));
    }
    (model_id, SnapshotQualifier::Undated)
}

fn compact_date(token: &str) -> Option<u32> {
    (token.len() == 8
        && token.bytes().all(|byte| byte.is_ascii_digit())
        && is_calendar_date(&token[..4], &token[4..6], &token[6..]))
    .then(|| token.parse().ok())
    .flatten()
}

fn dashed_date(token: &str) -> Option<u32> {
    let mut parts = token.split('-');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(year), Some(month), Some(day), None)
            if year.len() == 4 && month.len() == 2 && day.len() == 2 =>
        {
            compact_date(&format!("{year}{month}{day}"))
        }
        _ => None,
    }
}

fn is_calendar_date(year: &str, month: &str, day: &str) -> bool {
    let (Ok(year), Ok(month), Ok(day)) =
        (year.parse::<u16>(), month.parse::<u8>(), day.parse::<u8>())
    else {
        return false;
    };
    (2000..=2099).contains(&year) && (1..=12).contains(&month) && (1..=31).contains(&day)
}

#[must_use]
pub fn default_max_output_tokens(provider: &str, model_id: &str) -> Option<u32> {
    get_model_capabilities(provider, model_id).and_then(ModelCapabilities::default_output_tokens)
}

#[must_use]
pub const fn supported_model_capabilities() -> &'static [ModelCapabilities] {
    MODEL_CAPABILITIES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_family(model_id: &str) -> &str {
        split_snapshot(model_id).0
    }

    /// A row where reasoning is DEARER than output — `alibaba/qwen3-32b` in the
    /// live feed ($2.80/M output, $8.40/M reasoning). Reasoning tokens ride
    /// inside `output_tokens`, so the whole output band must bill at the higher
    /// reasoning rate, or the estimate under-bills and the cap can miss.
    #[test]
    fn output_bills_at_the_dearer_reasoning_rate() -> anyhow::Result<()> {
        use anyhow::Context;
        let pricing = Pricing::flat(0.7, 2.8).with_reasoning(8.4);
        let usage = Usage {
            served_speed: None,
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            request_input_tokens: None,
        };
        // 1M input @ $0.70 + 1M output @ max($2.80, $8.40) = 0.7 + 8.4 = 9.1.
        let cost = pricing.estimate_cost_usd(&usage).context("priced")?;
        assert!((cost - 9.1).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    /// When reasoning is CHEAPER than output the max leaves output unchanged —
    /// billing the whole band at the output rate (a bounded over-estimate for
    /// the reasoning tokens, the safe direction).
    #[test]
    fn output_keeps_the_dearer_output_rate() -> anyhow::Result<()> {
        use anyhow::Context;
        let pricing = Pricing::flat(1.0, 8.0).with_reasoning(3.0);
        let usage = Usage {
            served_speed: None,
            input_tokens: 0,
            output_tokens: 1_000_000,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            request_input_tokens: None,
        };
        // 1M output @ max($8, $3) = $8.
        let cost = pricing.estimate_cost_usd(&usage).context("priced")?;
        assert!((cost - 8.0).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    #[test]
    fn test_lookup_anthropic_fable_5() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps = get_model_capabilities("anthropic", "claude-fable-5")
            .context("claude-fable-5 capabilities missing")?;
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        assert!(caps.supports_adaptive_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
        let pricing = caps.pricing.context("pricing missing")?;
        let input = pricing.input.context("input price missing")?;
        let output = pricing.output.context("output price missing")?;
        assert!((input.usd_per_million_tokens - 10.0).abs() < f64::EPSILON);
        assert!((output.usd_per_million_tokens - 50.0).abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn test_lookup_anthropic_fable_51() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps = get_model_capabilities("anthropic", "claude-fable-5-1")
            .context("claude-fable-5-1 capabilities missing")?;
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        assert!(caps.supports_adaptive_thinking);
        assert!(caps.rejects_budget_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
        let pricing = caps.pricing.context("pricing missing")?;
        assert_eq!(pricing.input, Some(PricePoint::new(10.0)));
        assert_eq!(pricing.output, Some(PricePoint::new(50.0)));
        assert_eq!(pricing.cached_input, Some(PricePoint::new(0.25)));
        Ok(())
    }

    #[test]
    fn fable_51_bills_cache_reads_at_a_quarter_of_the_usual_rate() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps = get_model_capabilities("anthropic", "claude-fable-5-1")
            .context("claude-fable-5-1 capabilities missing")?;
        let usage = Usage {
            served_speed: None,
            input_tokens: 1_000_000,
            output_tokens: 0,
            cached_input_tokens: 1_000_000,
            cache_creation_input_tokens: 0,
            request_input_tokens: None,
        };
        let cost = caps.estimate_cost_usd(&usage).context("priced")?;
        assert!((cost - 0.25).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    #[test]
    fn test_lookup_anthropic_opus_55() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps =
            get_model_capabilities("anthropic", "claude-opus-5-5").context("opus 5.5 missing")?;
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        assert!(caps.supports_adaptive_thinking);
        assert!(caps.rejects_budget_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
        let pricing = caps.pricing.context("pricing missing")?;
        let input = pricing.input.context("input price missing")?;
        let output = pricing.output.context("output price missing")?;
        assert!((input.usd_per_million_tokens - 4.0).abs() < f64::EPSILON);
        assert!((output.usd_per_million_tokens - 20.0).abs() < f64::EPSILON);
        assert_eq!(pricing.cached_input, Some(PricePoint::new(0.2)));
        assert_eq!(pricing.cache_write, Some(PricePoint::new(5.0)));
        Ok(())
    }

    #[test]
    fn test_lookup_anthropic_haiku_55() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps =
            get_model_capabilities("anthropic", "claude-haiku-5-5").context("haiku 5.5 missing")?;
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        assert!(caps.supports_adaptive_thinking);
        assert!(caps.rejects_budget_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
        let pricing = caps.pricing.context("pricing missing")?;
        assert_eq!(pricing.input, Some(PricePoint::new(0.1)));
        assert_eq!(pricing.output, Some(PricePoint::new(0.5)));
        assert_eq!(pricing.cached_input, Some(PricePoint::new(0.01)));
        assert_eq!(pricing.cache_write, Some(PricePoint::new(0.125)));
        Ok(())
    }

    #[test]
    fn haiku_55_prices_each_input_band_at_its_own_rate() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps =
            get_model_capabilities("anthropic", "claude-haiku-5-5").context("haiku 5.5 missing")?;
        let usage = Usage {
            served_speed: None,
            input_tokens: 3_000_000,
            output_tokens: 1_000_000,
            cached_input_tokens: 1_000_000,
            cache_creation_input_tokens: 1_000_000,
            request_input_tokens: None,
        };
        // 1M plain @ $0.10 + 1M cache read @ $0.01 + 1M cache write @ $0.125
        // + 1M output @ $0.50.
        let cost = caps.estimate_cost_usd(&usage).context("priced")?;
        assert!((cost - 0.735).abs() < 1e-9, "unexpected cost: {cost}");
        Ok(())
    }

    #[test]
    fn test_lookup_anthropic_opus_5() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps = get_model_capabilities("anthropic", "claude-opus-5")
            .context("claude-opus-5 capabilities missing")?;
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        // Like Opus 4.8: manual budget_tokens 400s; adaptive supported, not required.
        assert!(caps.supports_adaptive_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
        let pricing = caps.pricing.context("pricing missing")?;
        let input = pricing.input.context("input price missing")?;
        let output = pricing.output.context("output price missing")?;
        assert!((input.usd_per_million_tokens - 5.0).abs() < f64::EPSILON);
        assert!((output.usd_per_million_tokens - 25.0).abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn test_lookup_anthropic_opus_48() {
        let caps = get_model_capabilities("anthropic", "claude-opus-4-8").unwrap();
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        assert!(caps.supports_adaptive_thinking);
    }

    #[test]
    fn test_lookup_anthropic_opus_46() {
        let caps = get_model_capabilities("anthropic", "claude-opus-4-6").unwrap();
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_adaptive_thinking);
    }

    #[test]
    fn test_lookup_anthropic_sonnet_55() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps = get_model_capabilities("anthropic", "claude-sonnet-5-5")
            .context("sonnet 5.5 missing")?;
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_adaptive_thinking);
        assert!(caps.rejects_budget_thinking);
        let pricing = caps.pricing.context("pricing missing")?;
        let input = pricing.input.context("input price missing")?;
        let output = pricing.output.context("output price missing")?;
        assert!((input.usd_per_million_tokens - 2.0).abs() < f64::EPSILON);
        assert!((output.usd_per_million_tokens - 10.0).abs() < f64::EPSILON);
        assert_eq!(pricing.cached_input, Some(PricePoint::new(0.1)));
        assert_eq!(pricing.cache_write, Some(PricePoint::new(2.5)));
        Ok(())
    }

    #[test]
    fn test_lookup_anthropic_sonnet_5() {
        let caps = get_model_capabilities("anthropic", "claude-sonnet-5").unwrap();
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        // Like Opus 4.8: manual budget_tokens 400s; adaptive supported, not required.
        assert!(caps.supports_adaptive_thinking);
    }

    #[test]
    fn test_lookup_anthropic_sonnet_46() {
        let caps = get_model_capabilities("anthropic", "claude-sonnet-4-6").unwrap();
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_adaptive_thinking);
    }

    #[test]
    fn openai_output_ceilings_match_the_model_pages() -> anyhow::Result<()> {
        use anyhow::Context as _;

        for (model, window, max_output) in [
            ("gpt-4.1", 1_000_000, 32_768),
            ("gpt-4.1-mini", 1_000_000, 32_768),
            ("gpt-4.1-nano", 1_000_000, 32_768),
            ("o1-mini", 128_000, 65_536),
        ] {
            let caps = get_model_capabilities("openai", model)
                .with_context(|| format!("{model} capabilities missing"))?;
            assert_eq!(
                (caps.context_window, caps.max_output_tokens),
                (Some(window), Some(max_output)),
                "{model}",
            );
        }
        Ok(())
    }

    #[test]
    fn default_reasoning_follows_the_vendor_docs() -> anyhow::Result<()> {
        use anyhow::Context as _;

        use DefaultReasoning::{Off, On};

        for (provider, model, expected) in [
            ("anthropic", "claude-opus-5-5", On),
            ("anthropic", "claude-fable-5-1", On),
            ("anthropic", "claude-sonnet-5-5", On),
            ("anthropic", "claude-haiku-5-5", On),
            ("anthropic", "claude-opus-4-8", Off),
            ("anthropic", "claude-sonnet-4-5-20250929", Off),
            ("openai", "o3", On),
            ("openai", "o1-mini", On),
            ("openai", "gpt-5", On),
            ("openai", "gpt-6.1-sol", On),
            ("openai", "gpt-5.6", On),
            ("openai", "gpt-5.4", Off),
            ("openai", "gpt-4.1", Off),
            ("openai", "z-ai/glm-5.1", On),
            ("openai", "glm-5", On),
            ("openai", "moonshotai/kimi-k3", On),
            ("openai", "minimax/minimax-m2.5", On),
            ("openai", "deepseek/deepseek-v4-pro", Off),
            ("openai", "accounts/fireworks/models/kimi-k3", On),
            ("openai", "kimi-k2.5", On),
            ("gemini", "gemini-2.5-pro", On),
            ("gemini", "gemini-3.1-pro-preview", On),
            ("gemini", "gemini-4-argon", On),
            ("gemini", "gemini-3.8-flash", On),
            ("gemini", "gemini-3.5-flash-lite", On),
        ] {
            let caps = get_model_capabilities(provider, model)
                .with_context(|| format!("{provider}/{model} capabilities missing"))?;
            assert_eq!(caps.default_reasoning, expected, "{provider}/{model}");
        }
        Ok(())
    }

    #[test]
    fn test_lookup_anthropic_sonnet_45_disables_adaptive_thinking() {
        let caps = get_model_capabilities("anthropic", "claude-sonnet-4-5-20250929").unwrap();
        assert!(!caps.supports_adaptive_thinking);
    }

    #[test]
    fn budget_rejecting_anthropic_models_carry_the_flag() {
        for model in [
            "claude-haiku-5-5",
            "claude-fable-5-1",
            "claude-fable-5",
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-5-5",
            "claude-sonnet-5",
            "claude-sonnet-4-6",
        ] {
            let caps = get_model_capabilities("anthropic", model).unwrap();
            assert!(
                caps.rejects_budget_thinking,
                "{model} rejects budget thinking and must carry the flag"
            );
        }
    }

    #[test]
    fn rejects_budget_thinking_implies_adaptive_is_available() {
        for caps in MODEL_CAPABILITIES {
            if caps.rejects_budget_thinking {
                assert!(
                    caps.supports_adaptive_thinking,
                    "{}/{} rejects budget thinking but does not support adaptive — \
                     callers would have no thinking-object shape to promote to",
                    caps.provider, caps.model_id,
                );
            }
        }
    }

    #[test]
    fn budget_capable_models_do_not_carry_the_flag() -> anyhow::Result<()> {
        use anyhow::Context as _;

        for (provider, model) in [
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("anthropic", "claude-haiku-4-5-20251001"),
            ("openai", "gpt-5.6-sol"),
            ("openai", "gpt-6-astra"),
            ("openai", "gpt-6-sol"),
            ("openai", "gpt-6.1-sol"),
            ("openai", "gpt-6-luna"),
        ] {
            let caps = get_model_capabilities(provider, model)
                .with_context(|| format!("{provider}/{model} capabilities missing"))?;
            assert!(
                !caps.rejects_budget_thinking,
                "{provider}/{model} accepts budget-style thinking config"
            );
        }
        Ok(())
    }

    #[test]
    fn test_lookup_openai_pricing() {
        let caps = get_model_capabilities("openai", "gpt-4o").unwrap();
        let pricing = caps.pricing.unwrap();
        assert!((pricing.input.unwrap().usd_per_million_tokens - 1.25).abs() < f64::EPSILON);
        assert!((pricing.output.unwrap().usd_per_million_tokens - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lookup_openai_gpt54() {
        let caps = get_model_capabilities("openai", "gpt-5.4").unwrap();
        assert_eq!(caps.context_window, Some(1_050_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
    }

    #[test]
    fn test_lookup_openai_gpt52_pro() -> anyhow::Result<()> {
        use anyhow::Context;

        let caps = get_model_capabilities("openai", "gpt-5.2-pro")
            .context("gpt-5.2-pro capabilities missing")?;
        assert_eq!(caps.context_window, Some(400_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
        let pricing = caps.pricing.context("gpt-5.2-pro pricing missing")?;
        let input = pricing.input.context("input price missing")?;
        let output = pricing.output.context("output price missing")?;
        assert!((input.usd_per_million_tokens - 21.0).abs() < f64::EPSILON);
        assert!((output.usd_per_million_tokens - 168.0).abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn test_lookup_openai_gpt56_and_gpt6_families() -> anyhow::Result<()> {
        use anyhow::Context as _;

        for (model_id, input, cached_input, output, cache_write_note) in [
            ("gpt-6-astra", 10.0, 1.0, 50.0, "$12.50/M"),
            ("gpt-6-sol", 2.0, 0.2, 10.0, "$2.50/M"),
            ("gpt-6.1-sol", 2.0, 0.1, 10.0, "$2.50/M"),
            ("gpt-6-luna", 0.1, 0.01, 0.5, "$0.125/M"),
            ("gpt-5.6", 5.0, 0.5, 30.0, "$6.25/M"),
            ("gpt-5.6-sol", 5.0, 0.5, 30.0, "$6.25/M"),
            ("gpt-5.6-terra", 2.5, 0.25, 15.0, "$3.125/M"),
            ("gpt-5.6-luna", 1.0, 0.1, 6.0, "$1.25/M"),
        ] {
            let caps = get_model_capabilities("openai", model_id)
                .with_context(|| format!("{model_id} capabilities missing"))?;
            assert_eq!(caps.context_window, Some(1_050_000));
            assert_eq!(caps.max_output_tokens, Some(128_000));
            assert!(caps.supports_thinking);
            assert!(caps.supports_adaptive_thinking);
            assert_eq!(caps.source_status, SourceStatus::Official);
            let pricing = caps
                .pricing
                .with_context(|| format!("{model_id} pricing missing"))?;
            assert_eq!(pricing.input, Some(PricePoint::new(input)));
            assert_eq!(pricing.cached_input, Some(PricePoint::new(cached_input)));
            assert_eq!(pricing.output, Some(PricePoint::new(output)));
            assert!(pricing.notes.is_some_and(|notes| {
                notes.contains(cache_write_note) && notes.contains("more than 272K")
            }));
        }

        Ok(())
    }

    #[test]
    fn gpt61_sol_bills_cached_input_at_half_the_gpt6_sol_rate() -> anyhow::Result<()> {
        use anyhow::Context as _;

        let usage = Usage {
            served_speed: None,
            input_tokens: 1_000_000,
            output_tokens: 0,
            cached_input_tokens: 1_000_000,
            cache_creation_input_tokens: 0,
            request_input_tokens: None,
        };
        for (model_id, expected) in [("gpt-6.1-sol", 0.1), ("gpt-6-sol", 0.2)] {
            let caps = get_model_capabilities("openai", model_id)
                .with_context(|| format!("{model_id} capabilities missing"))?;
            let cost = caps.estimate_cost_usd(&usage).context("priced")?;
            assert!(
                (cost - expected).abs() < 1e-9,
                "{model_id}: unexpected cost {cost}"
            );
        }
        let sol_61 = get_model_capabilities("openai", "gpt-6.1-sol")
            .context("gpt-6.1-sol capabilities missing")?
            .pricing
            .context("gpt-6.1-sol pricing missing")?;
        assert_eq!(sol_61.cache_write, Some(PricePoint::new(2.5)));
        Ok(())
    }

    #[test]
    fn test_lookup_openai_gpt53_codex() {
        let caps = get_model_capabilities("openai", "gpt-5.3-codex").unwrap();
        assert_eq!(caps.context_window, Some(400_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_adaptive_thinking);
        assert!(caps.supports_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
    }

    #[test]
    fn gemini_4_argon_row_matches_the_launch_docs() -> anyhow::Result<()> {
        use anyhow::Context as _;

        let argon = get_model_capabilities("gemini", "gemini-4-argon")
            .context("gemini-4-argon capabilities missing")?;
        assert_eq!(argon.context_window, Some(1_048_576));
        assert_eq!(argon.max_output_tokens, Some(65_536));
        assert_eq!(argon.default_output_tokens(), Some(65_536));
        assert!(argon.supports_thinking);
        assert!(!argon.supports_adaptive_thinking);
        assert!(!argon.rejects_budget_thinking);
        assert_eq!(argon.default_reasoning, DefaultReasoning::On);
        assert_eq!(argon.source_status, SourceStatus::Official);

        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cached_input_tokens: 500_000,
            cache_creation_input_tokens: 0,
            served_speed: None,
            request_input_tokens: None,
        };
        let cost = argon
            .estimate_cost_usd(&usage)
            .context("gemini-4-argon is priced")?;
        assert!((cost - (1.0 + 0.05 + 10.0)).abs() < 1e-9, "{cost}");
        Ok(())
    }

    #[test]
    fn gemini_catalog_lists_the_current_lineup() -> anyhow::Result<()> {
        use anyhow::Context as _;

        for (model, input, output, cached) in [
            ("gemini-3.8-flash", 0.75, 3.75, 0.075),
            ("gemini-3.6-flash", 0.75, 3.75, 0.075),
            ("gemini-3.5-flash-lite", 0.30, 2.50, 0.03),
            ("gemini-3.1-flash-lite", 0.25, 1.50, 0.025),
        ] {
            let caps = get_model_capabilities("gemini", model)
                .with_context(|| format!("{model} capabilities missing"))?;
            assert_eq!(
                (caps.context_window, caps.max_output_tokens),
                (Some(1_048_576), Some(65_536)),
                "{model}",
            );
            assert!(caps.supports_thinking, "{model}");
            let pricing = caps.pricing.with_context(|| format!("{model} unpriced"))?;
            let rate = |point: Option<PricePoint>| point.map(|p| p.usd_per_million_tokens);
            assert_eq!(rate(pricing.input), Some(input), "{model}");
            assert_eq!(rate(pricing.output), Some(output), "{model}");
            assert_eq!(rate(pricing.cached_input), Some(cached), "{model}");
        }

        for redirected in ["gemini-3.7-flash", "gemini-3.5-flash"] {
            let caps = get_model_capabilities("gemini", redirected)
                .with_context(|| format!("{redirected} must stay resolvable"))?;
            assert_eq!(caps.max_output_tokens, Some(65_536), "{redirected}");
        }

        for shut_down in [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.0-pro",
        ] {
            assert!(
                get_model_capabilities("gemini", shut_down).is_none(),
                "{shut_down} is shut down"
            );
        }
        Ok(())
    }

    #[test]
    fn test_lookup_gemini_preview_models() {
        let flash = get_model_capabilities("gemini", "gemini-3-flash-preview").unwrap();
        assert_eq!(flash.context_window, Some(1_048_576));
        assert!(flash.supports_thinking);

        let pro = get_model_capabilities("gemini", "gemini-3.1-pro-preview").unwrap();
        assert_eq!(pro.max_output_tokens, Some(65_536));
        assert!(pro.supports_thinking);
    }

    #[test]
    fn test_lookup_open_reasoning_models_resolve_with_thinking() {
        // DeepSeek V4 Pro via OpenRouter slug — reasoning model.
        let deepseek = get_model_capabilities("openai", "deepseek/deepseek-v4-pro").unwrap();
        assert!(deepseek.supports_thinking);
        assert_eq!(deepseek.max_output_tokens, Some(384_000));
        let pricing = deepseek.pricing.unwrap();
        assert!(pricing.input.unwrap().usd_per_million_tokens > 0.0);
        assert!(pricing.output.unwrap().usd_per_million_tokens > 0.0);

        // z.ai GLM-5.1 via OpenRouter slug — reasoning model.
        let glm = get_model_capabilities("openai", "z-ai/glm-5.1").unwrap();
        assert!(glm.supports_thinking);
        assert_eq!(glm.max_output_tokens, Some(131_072));
        let glm_pricing = glm.pricing.unwrap();
        assert!((glm_pricing.input.unwrap().usd_per_million_tokens - 0.98).abs() < f64::EPSILON);
        assert!((glm_pricing.output.unwrap().usd_per_million_tokens - 3.08).abs() < f64::EPSILON);

        // Kimi K2 Thinking native — reasoning model.
        let kimi_thinking = get_model_capabilities("openai", "kimi-k2-thinking").unwrap();
        assert!(kimi_thinking.supports_thinking);
        assert_eq!(kimi_thinking.max_output_tokens, Some(131_072));
        assert!(
            kimi_thinking
                .pricing
                .unwrap()
                .output
                .unwrap()
                .usd_per_million_tokens
                > 0.0
        );
    }

    #[test]
    fn test_lookup_open_non_reasoning_kimi_models() -> anyhow::Result<()> {
        use anyhow::Context as _;

        // Kimi K2.6 / K2.5 are registered as non-reasoning coding models.
        let k26 = get_model_capabilities("openai", "moonshotai/kimi-k2.6")
            .context("moonshotai/kimi-k2.6 capabilities missing")?;
        assert!(!k26.supports_thinking);
        assert_eq!(k26.max_output_tokens, None);
        assert_eq!(k26.default_output_tokens(), Some(65_536));
        let input = k26
            .pricing
            .and_then(|pricing| pricing.input)
            .context("moonshotai/kimi-k2.6 input pricing missing")?;
        assert!(input.usd_per_million_tokens > 0.0);

        let k25 = get_model_capabilities("openai", "moonshotai/kimi-k2.5")
            .context("moonshotai/kimi-k2.5 capabilities missing")?;
        assert_eq!(k25.max_output_tokens, None);
        assert_eq!(k25.default_output_tokens(), Some(32_768));
        assert_eq!(
            default_max_output_tokens("openai", "moonshotai/Kimi-K2.5"),
            Some(32_768)
        );

        let k25_native = get_model_capabilities("openai", "kimi-k2.5")
            .context("kimi-k2.5 capabilities missing")?;
        assert!(!k25_native.supports_thinking);
        assert_eq!(k25_native.max_output_tokens, Some(32_768));
        Ok(())
    }

    #[test]
    fn test_lookup_kimi_k3_routes() -> anyhow::Result<()> {
        use anyhow::Context as _;

        let k3 = get_model_capabilities("openai", "moonshotai/kimi-k3")
            .context("missing capabilities for moonshotai/kimi-k3")?;
        assert!(k3.supports_thinking);
        assert_eq!(k3.context_window, Some(1_048_576));
        assert_eq!(k3.max_output_tokens, None);
        assert!(k3.pricing.is_none());

        let fireworks = get_model_capabilities("openai", "accounts/fireworks/models/kimi-k3")
            .context("missing capabilities for accounts/fireworks/models/kimi-k3")?;
        assert_eq!(fireworks.context_window, Some(1_048_576));
        assert_eq!(fireworks.max_output_tokens, None);
        assert!(fireworks.pricing.is_none());
        assert!(fireworks.supports_thinking);

        let baseten = get_model_capabilities("openai", "moonshotai/Kimi-K3")
            .context("missing capabilities for moonshotai/Kimi-K3")?;
        assert_eq!(baseten.model_id, "moonshotai/kimi-k3");
        assert_eq!(baseten.max_output_tokens, None);
        Ok(())
    }

    #[test]
    fn test_lookup_all_open_models_resolve() {
        // Every model_id below is exactly how the consumer looks them up
        // (provider == "openai" for all open routes).
        for model_id in [
            "z-ai/glm-5.1",
            "glm-5",
            "moonshotai/kimi-k2.6",
            "moonshotai/kimi-k2.5",
            "kimi-k2.5",
            "kimi-k2-thinking",
            "deepseek/deepseek-v4-pro",
            "deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
            "deepseek-v4-flash",
            "MiniMax-M2.5",
            "minimax/minimax-m2.5",
        ] {
            let caps = get_model_capabilities("openai", model_id)
                .unwrap_or_else(|| panic!("missing capabilities for {model_id}"));
            assert!(
                caps.pricing.is_some(),
                "pricing should be populated for {model_id}"
            );
            assert!(
                caps.default_output_tokens().is_some_and(|m| m > 0),
                "the default output budget should be non-zero for {model_id}"
            );
            assert!(
                caps.context_window.is_some_and(|c| c > 0),
                "context_window should be non-zero for {model_id}"
            );
        }
    }

    #[test]
    fn test_lookup_minimax_native_pricing() {
        let native = get_model_capabilities("openai", "MiniMax-M2.5").unwrap();
        assert!(native.supports_thinking);
        let pricing = native.pricing.unwrap();
        assert!((pricing.input.unwrap().usd_per_million_tokens - 0.3).abs() < f64::EPSILON);
        assert!((pricing.output.unwrap().usd_per_million_tokens - 1.2).abs() < f64::EPSILON);
        // Cache-read is the first-party platform.minimax.io PAYG rate ($0.03/M),
        // not the ~$0.155/M that an earlier entry overstated by ~3-5x.
        assert!((pricing.cached_input.unwrap().usd_per_million_tokens - 0.03).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_cost_usd() {
        let caps = get_model_capabilities("openai", "gpt-4o").unwrap();
        let cost = caps
            .estimate_cost_usd(&Usage {
                served_speed: None,
                input_tokens: 2_000,
                output_tokens: 1_000,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                request_input_tokens: None,
            })
            .unwrap();
        assert!((cost - 0.0075).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_cost_usd_with_cached_input() {
        let caps = get_model_capabilities("openai", "gpt-5.4").unwrap();
        let cost = caps
            .estimate_cost_usd(&Usage {
                served_speed: None,
                input_tokens: 2_000,
                output_tokens: 1_000,
                cached_input_tokens: 1_000,
                cache_creation_input_tokens: 0,
                request_input_tokens: None,
            })
            .unwrap();
        assert!((cost - 0.01775).abs() < f64::EPSILON);
    }

    /// Every way a provider or a host spells a registered model — its
    /// undated alias, a dated snapshot, a Google Cloud `@` version or a
    /// `-latest` alias — resolves to that model's row, and no two rows of
    /// one provider share a snapshot family, so the resolution is unique.
    #[test]
    fn every_snapshot_spelling_of_a_registered_model_resolves_to_its_row() -> anyhow::Result<()> {
        use anyhow::Context;

        for row in MODEL_CAPABILITIES {
            let family = snapshot_family(row.model_id);
            let twins: Vec<&str> = MODEL_CAPABILITIES
                .iter()
                .filter(|other| {
                    other.provider == row.provider && snapshot_family(other.model_id) == family
                })
                .map(|other| other.model_id)
                .collect();
            assert_eq!(
                twins,
                [row.model_id],
                "{} rows share family {family}",
                row.provider
            );

            let mut dates: Vec<u32> = SNAPSHOTS_OF_UNDATED_ROWS
                .iter()
                .filter(|(listed, _)| *listed == row.model_id)
                .filter_map(|(_, date)| dashed_date(date))
                .collect();
            if let (_, SnapshotQualifier::Dated(date)) = split_snapshot(row.model_id) {
                dates.push(date);
            }
            let mut spellings = vec![
                row.model_id.to_owned(),
                row.model_id.to_ascii_uppercase(),
                family.to_owned(),
                format!("{family}-latest"),
            ];
            for date in dates {
                let (year, month, day) = (date / 10_000, date / 100 % 100, date % 100);
                spellings.push(format!("{family}-{date}"));
                spellings.push(format!("{family}-{year}-{month:02}-{day:02}"));
                spellings.push(format!("{family}@{date}"));
            }
            for spelling in spellings {
                let caps = get_model_capabilities(row.provider, &spelling)
                    .with_context(|| format!("{}/{spelling} resolves to nothing", row.provider))?;
                assert_eq!(caps.model_id, row.model_id, "{}/{spelling}", row.provider);
            }
            for unlisted in [
                format!("{family}-20991231"),
                format!("{family}-2099-12-31"),
                format!("{family}@20991231"),
                format!("{family}@002"),
            ] {
                assert!(
                    get_model_capabilities(row.provider, &unlisted).is_none(),
                    "{}/{unlisted} names a snapshot no row describes",
                    row.provider
                );
            }
        }
        Ok(())
    }

    #[test]
    fn every_listed_snapshot_names_exactly_one_undated_row() {
        for (row_id, date) in SNAPSHOTS_OF_UNDATED_ROWS {
            assert!(
                dashed_date(date).is_some(),
                "{row_id}: {date} is not a YYYY-MM-DD calendar date"
            );
            let rows: Vec<&str> = MODEL_CAPABILITIES
                .iter()
                .filter(|caps| caps.model_id == *row_id)
                .map(|caps| caps.model_id)
                .collect();
            assert_eq!(
                rows.len(),
                1,
                "{row_id} must name exactly one row: {rows:?}"
            );
            assert_eq!(
                split_snapshot(row_id).1,
                SnapshotQualifier::Undated,
                "{row_id} must be an undated row"
            );
        }
    }

    #[test]
    fn snapshots_with_their_own_limits_stay_unknown() {
        for (provider, spelling) in [
            ("openai", "gpt-4o-2024-05-13"),
            ("anthropic", "claude-3-5-sonnet-20240620"),
        ] {
            assert!(
                get_model_capabilities(provider, spelling).is_none(),
                "{provider}/{spelling} must not borrow another snapshot's limits"
            );
            assert_eq!(default_max_output_tokens(provider, spelling), None);
        }
    }

    #[test]
    fn opus_45_aliases_resolve_to_the_dated_snapshot() -> anyhow::Result<()> {
        use anyhow::Context;

        for spelling in [
            "claude-opus-4-5-20251101",
            "claude-opus-4-5",
            "claude-opus-4-5@20251101",
        ] {
            let caps =
                get_model_capabilities("anthropic", spelling).context("opus 4.5 is unknown")?;
            assert_eq!(caps.model_id, "claude-opus-4-5-20251101");
            assert_eq!(caps.context_window, Some(200_000));
            assert_eq!(caps.max_output_tokens, Some(64_000));
            assert!(caps.supports_thinking);
            assert!(!caps.supports_adaptive_thinking);
            assert!(!caps.rejects_budget_thinking);
            let pricing = caps.pricing.context("pricing missing")?;
            assert_eq!(pricing.input, Some(PricePoint::new(5.0)));
            assert_eq!(pricing.output, Some(PricePoint::new(25.0)));
            assert_eq!(pricing.cached_input, Some(PricePoint::new(0.5)));
            assert_eq!(pricing.cache_write, Some(PricePoint::new(6.25)));
        }
        Ok(())
    }

    #[test]
    fn dated_snapshots_of_undated_rows_resolve() -> anyhow::Result<()> {
        use anyhow::Context;

        for (provider, spelling, expected) in [
            ("openai", "gpt-5.5-2026-04-23", "gpt-5.5"),
            ("openai", "gpt-4o-2024-08-06", "gpt-4o"),
            (
                "anthropic",
                "claude-sonnet-4-5",
                "claude-sonnet-4-5-20250929",
            ),
            ("anthropic", "claude-haiku-4-5", "claude-haiku-4-5-20251001"),
            (
                "anthropic",
                "claude-haiku-4-5@20251001",
                "claude-haiku-4-5-20251001",
            ),
            (
                "anthropic",
                "claude-3-5-haiku-latest",
                "claude-3-5-haiku-20241022",
            ),
        ] {
            let caps = get_model_capabilities(provider, spelling)
                .with_context(|| format!("{provider}/{spelling} is unknown"))?;
            assert_eq!(caps.model_id, expected, "{provider}/{spelling}");
        }
        Ok(())
    }

    #[test]
    fn only_calendar_dates_are_snapshot_qualifiers() {
        for (spelling, family) in [
            ("claude-opus-4-5-20251301", "claude-opus-4-5-20251301"),
            ("claude-opus-4-5-2025110", "claude-opus-4-5-2025110"),
            ("gpt-5.5-2026-13-01", "gpt-5.5-2026-13-01"),
            ("gpt-5-mini", "gpt-5-mini"),
            ("o3-mini", "o3-mini"),
            ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
            (
                "accounts/fireworks/models/kimi-k3",
                "accounts/fireworks/models/kimi-k3",
            ),
        ] {
            assert_eq!(snapshot_family(spelling), family, "{spelling}");
        }
        assert!(get_model_capabilities("anthropic", "claude-opus-4-5-20251301").is_none());
        assert!(get_model_capabilities("anthropic", "claude-opus-4-1-20250805").is_none());
    }
}
