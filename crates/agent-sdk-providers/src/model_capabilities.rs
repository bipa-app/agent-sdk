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
    pub cached_input: Option<PricePoint>,
    pub notes: Option<&'static str>,
}

impl Pricing {
    #[must_use]
    pub const fn flat(input: f64, output: f64) -> Self {
        Self {
            input: Some(PricePoint::new(input)),
            output: Some(PricePoint::new(output)),
            cached_input: None,
            notes: None,
        }
    }

    #[must_use]
    pub const fn flat_with_cached(input: f64, output: f64, cached_input: f64) -> Self {
        Self {
            input: Some(PricePoint::new(input)),
            output: Some(PricePoint::new(output)),
            cached_input: Some(PricePoint::new(cached_input)),
            notes: None,
        }
    }

    #[must_use]
    pub const fn with_notes(mut self, notes: &'static str) -> Self {
        self.notes = Some(notes);
        self
    }

    #[must_use]
    pub fn estimate_cost_usd(&self, usage: &Usage) -> Option<f64> {
        let cached_input_tokens = usage.cached_input_tokens.min(usage.input_tokens);
        let uncached_input_tokens = usage.input_tokens.saturating_sub(cached_input_tokens);

        let input = match (self.input, self.cached_input) {
            (Some(input), Some(cached_input)) => Some(
                input.estimate_cost_usd(uncached_input_tokens)
                    + cached_input.estimate_cost_usd(cached_input_tokens),
            ),
            (Some(input), None) => Some(input.estimate_cost_usd(usage.input_tokens)),
            (None, Some(cached_input)) => Some(cached_input.estimate_cost_usd(cached_input_tokens)),
            (None, None) => None,
        };
        let output = self
            .output
            .map(|p| p.estimate_cost_usd(usage.output_tokens));
        match (input, output) {
            (Some(input), Some(output)) => Some(input + output),
            (Some(input), None) => Some(input),
            (None, Some(output)) => Some(output),
            (None, None) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelCapabilities {
    pub provider: &'static str,
    pub model_id: &'static str,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub pricing: Option<Pricing>,
    pub supports_thinking: bool,
    pub supports_adaptive_thinking: bool,
    pub source_url: &'static str,
    pub source_status: SourceStatus,
    pub notes: Option<&'static str>,
}

impl ModelCapabilities {
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
const OPENAI_GPT56_SOL_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-sol";
const OPENAI_GPT56_TERRA_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-terra";
const OPENAI_GPT56_LUNA_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-luna";
const OPENAI_GPT54_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.4";
const OPENAI_GPT53_CODEX_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.3-codex";
const OPENAI_GPT52_PRO_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.2-pro";
const GOOGLE_MODELS_URL: &str = "https://ai.google.dev/gemini-api/docs/models";
const GOOGLE_PRICING_URL: &str = "https://ai.google.dev/gemini-api/docs/pricing";

// Open-model routes. All reached through OpenAIProvider (provider()=="openai"),
// whether via OpenRouter slugs or the native z.ai / Moonshot / MiniMax base URLs.
const OPENROUTER_GLM51_URL: &str = "https://openrouter.ai/z-ai/glm-5.1";
const ZAI_GLM5_PRICING_URL: &str = "https://docs.z.ai/guides/overview/pricing";
const OPENROUTER_KIMI_K26_URL: &str = "https://openrouter.ai/moonshotai/kimi-k2.6";
const OPENROUTER_KIMI_K25_URL: &str = "https://openrouter.ai/moonshotai/kimi-k2.5";
const KIMI_K25_AA_URL: &str = "https://artificialanalysis.ai/models/kimi-k2-5";
const OPENROUTER_KIMI_K2_THINKING_URL: &str = "https://openrouter.ai/moonshotai/kimi-k2-thinking";
const OPENROUTER_DEEPSEEK_V4_PRO_URL: &str = "https://openrouter.ai/deepseek/deepseek-v4-pro";
const OPENROUTER_DEEPSEEK_V4_FLASH_URL: &str = "https://openrouter.ai/deepseek/deepseek-v4-flash";
const DEEPSEEK_PRICING_URL: &str = "https://api-docs.deepseek.com/quick_start/pricing";
const MINIMAX_PRICING_URL: &str = "https://platform.minimax.io/docs/guides/pricing-paygo";
const OPENROUTER_MINIMAX_M25_URL: &str = "https://openrouter.ai/minimax/minimax-m2.5";

const MODEL_CAPABILITIES: &[ModelCapabilities] = &[
    // Anthropic
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-fable-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat(10.0, 50.0).with_notes("Anthropic Fable 5 official pricing: $10 input / $50 output per 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Fable 5 is adaptive-only: adaptive thinking is always on (applies even when `thinking` is unset) and `ThinkingMode::Enabled { budget_tokens }` is rejected by the Anthropic API. The SDK fails fast in validate_thinking_config. Raw chain of thought is never returned — thinking blocks arrive empty (the SDK requests thinking display=omitted). Safety classifiers may decline a request with stop_reason=refusal on an HTTP 200."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-8",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat(5.0, 25.0).with_notes("Anthropic Opus 4.8 pricing matches the Opus 4.6 tier ($5/$25 per 1M); verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Opus 4.8 requires adaptive thinking — `ThinkingMode::Enabled { budget_tokens }` is rejected by the Anthropic API. The SDK fails fast in validate_thinking_config."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-7",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat(5.0, 25.0).with_notes("Anthropic Opus 4.7 pricing matches the Opus 4.6 tier ($5/$25 per 1M); verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Opus 4.7 requires adaptive thinking — `ThinkingMode::Enabled { budget_tokens }` is rejected by the Anthropic API. The SDK fails fast in validate_thinking_config."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-6",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat(5.0, 25.0).with_notes("Anthropic Opus 4.6 pricing from bundled Claude API guidance; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Current Anthropic docs show this model alongside 200K/128K markers."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-5",
        context_window: Some(1_000_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet 5 standard pricing $3/$15 per 1M; introductory $2/$10 through 2026-08-31. A new tokenizer produces ~30% more tokens than Sonnet 4.6, so equivalent-text cost differs even at unchanged per-token rates.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Official,
        notes: Some("Sonnet 5 is adaptive-only: adaptive thinking is on by default (applies even when `thinking` is unset) and `ThinkingMode::Enabled { budget_tokens }` returns 400 from the Anthropic API — same as Opus 4.8. Non-default sampling params (temperature/top_p/top_k) also return 400 (constraint inherited from Opus 4.7). Uses a new tokenizer (~30% more tokens than Sonnet 4.6)."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-4-6",
        context_window: Some(1_000_000),
        max_output_tokens: Some(64_000),
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Anthropic docs list Sonnet 4.6; user confirmed adaptive thinking support."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-4-5-20250929",
        context_window: Some(200_000),
        max_output_tokens: Some(64_000),
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-haiku-4-5-20251001",
        context_window: Some(200_000),
        max_output_tokens: Some(64_000),
        pricing: Some(Pricing::flat(1.0, 5.0).with_notes("Anthropic Haiku tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-4-20250514",
        context_window: Some(200_000),
        max_output_tokens: Some(64_000),
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-20250514",
        context_window: Some(200_000),
        max_output_tokens: Some(32_000),
        pricing: Some(Pricing::flat(15.0, 75.0).with_notes("Anthropic Opus tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-3-5-sonnet-20241022",
        context_window: Some(200_000),
        max_output_tokens: Some(8_192),
        pricing: Some(Pricing::flat(3.0, 15.0).with_notes("Anthropic Sonnet tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-3-5-haiku-20241022",
        context_window: Some(200_000),
        max_output_tokens: Some(8_192),
        pricing: Some(Pricing::flat(1.0, 5.0).with_notes("Anthropic Haiku tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: None,
    },
    // OpenAI
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.6",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(5.0, 30.0, 0.5).with_notes(
            "Standard tier base rates. Cache writes cost $6.25/M input tokens. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: OPENAI_GPT56_SOL_URL,
        source_status: SourceStatus::Official,
        notes: Some("Official alias for GPT-5.6 Sol."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.6-sol",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(5.0, 30.0, 0.5).with_notes(
            "Standard tier base rates. Cache writes cost $6.25/M input tokens. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: OPENAI_GPT56_SOL_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.6-terra",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(2.5, 15.0, 0.25).with_notes(
            "Standard tier base rates. Cache writes cost $3.125/M input tokens. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: OPENAI_GPT56_TERRA_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.6-luna",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(1.0, 6.0, 0.1).with_notes(
            "Standard tier base rates. Cache writes cost $1.25/M input tokens. Requests with more than 272K input tokens cost 2x input and 1.5x output for the full request.",
        )),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: OPENAI_GPT56_LUNA_URL,
        source_status: SourceStatus::Official,
        notes: Some("Supports Chat Completions and Responses, 1.05M context, and 128K max output."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.4",
        context_window: Some(1_050_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(2.50, 15.0, 0.25)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENAI_GPT54_URL,
        source_status: SourceStatus::Official,
        notes: Some("OpenAI model docs list 1.05M context, 128K max output, and reasoning.effort support."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.3-codex",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(1.50, 6.0, 0.375)),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: OPENAI_GPT53_CODEX_URL,
        source_status: SourceStatus::Official,
        notes: Some("OpenAI model docs list Responses-only access, a 272K maximum input, 128K maximum output, and reasoning.effort levels."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(1.25, 10.0, 0.125)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5-mini",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(0.125, 1.0, 0.0125)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5-nano",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat_with_cached(0.025, 0.20, 0.0025)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.2-instant",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        pricing: None,
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model exists in OpenAI docs, but pricing was not extracted from the official pricing page in this pass."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.2-thinking",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENAI_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model exists in OpenAI docs, but pricing was not extracted from the official pricing page in this pass."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.2-pro",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat(21.0, 168.0)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENAI_GPT52_PRO_URL,
        source_status: SourceStatus::Official,
        notes: Some("Responses-only pro model. Supports medium, high, and xhigh reasoning effort."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-5.2-codex",
        context_window: Some(400_000),
        max_output_tokens: Some(128_000),
        pricing: None,
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model presence confirmed from OpenAI docs; pricing not yet extracted in this pass."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o3",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        pricing: Some(Pricing::flat(1.0, 4.0)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o3-mini",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        pricing: Some(Pricing::flat(0.55, 2.20)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o4-mini",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        pricing: Some(Pricing::flat(0.55, 2.20)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o1",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        pricing: Some(Pricing::flat(7.50, 30.0)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "o1-mini",
        context_window: Some(200_000),
        max_output_tokens: Some(100_000),
        pricing: Some(Pricing::flat(0.55, 2.20)),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4.1",
        context_window: Some(1_000_000),
        max_output_tokens: Some(16_384),
        pricing: Some(Pricing::flat(1.0, 4.0)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context window from model family docs/notes."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4.1-mini",
        context_window: Some(1_000_000),
        max_output_tokens: Some(16_384),
        pricing: Some(Pricing::flat(0.20, 0.80)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context window from model family docs/notes."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4.1-nano",
        context_window: Some(1_000_000),
        max_output_tokens: Some(16_384),
        pricing: Some(Pricing::flat(0.05, 0.20)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context window from model family docs/notes."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4o",
        context_window: Some(128_000),
        max_output_tokens: Some(16_384),
        pricing: Some(Pricing::flat(1.25, 5.0)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output from existing runtime assumptions."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "gpt-4o-mini",
        context_window: Some(128_000),
        max_output_tokens: Some(16_384),
        pricing: Some(Pricing::flat(0.075, 0.30)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output from existing runtime assumptions."),
    },
    // Gemini
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.1-pro-preview",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        pricing: Some(Pricing::flat(2.0, 12.0).with_notes("Official pricing for prompts <= 200K tokens. For prompts > 200K, pricing increases to $4 input / $18 output per 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing sourced from Gemini 3.1 Pro Preview docs."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.1-pro",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        pricing: Some(Pricing::flat(2.0, 12.0).with_notes("Legacy alias retained for compatibility. For prompts > 200K, pricing increases to $4 input / $18 output per 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Legacy Gemini 3.1 Pro alias retained for compatibility; prefer gemini-3.1-pro-preview."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.1-flash-lite-preview",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model presence confirmed from Google docs, but pricing was not extracted in this pass."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3-flash-preview",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model presence confirmed from Google docs, but pricing was not extracted in this pass."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.0-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Legacy Gemini 3.0 Flash model retained for compatibility; prefer gemini-3-flash-preview."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-3.0-pro",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model presence confirmed from Google docs, but pricing was not extracted in this pass."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-2.5-flash",
        context_window: Some(1_000_000),
        max_output_tokens: Some(65_536),
        pricing: Some(Pricing::flat(0.30, 2.50).with_notes("Official text/image/video pricing. Audio input is priced separately at $1.00 / 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Official docs state output pricing includes thinking tokens."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-2.5-pro",
        context_window: Some(1_000_000),
        max_output_tokens: Some(65_536),
        pricing: None,
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_MODELS_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Model presence confirmed from Google docs, but pricing was not extracted in this pass."),
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-2.0-flash",
        context_window: Some(1_000_000),
        max_output_tokens: Some(8_192),
        pricing: Some(Pricing::flat(0.10, 0.40).with_notes("Official text/image/video pricing. Audio input is priced separately at $0.70 / 1M tokens.")),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: None,
    },
    ModelCapabilities {
        provider: "gemini",
        model_id: "gemini-2.0-flash-lite",
        context_window: Some(1_000_000),
        max_output_tokens: Some(8_192),
        pricing: Some(Pricing::flat(0.075, 0.30)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: None,
    },
    // Open models (z.ai / Moonshot / DeepSeek / MiniMax). All routed through
    // OpenAIProvider, so provider == "openai" and the model_id is the exact
    // string the caller passes (OpenRouter slug or native model id).
    ModelCapabilities {
        provider: "openai",
        model_id: "z-ai/glm-5.1",
        context_window: Some(202_752),
        max_output_tokens: Some(131_072),
        pricing: Some(Pricing::flat(0.98, 3.08).with_notes("OpenRouter rate for z-ai/glm-5.1: input $0.98/M, output $3.08/M.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENROUTER_GLM51_URL,
        source_status: SourceStatus::Derived,
        notes: Some("GLM-5.1 (z.ai/Zhipu) via OpenRouter slug. Reasoning/thinking model; context 203K (=202,752). max_output 128K from z.ai GLM-5.1 docs, sized generously for hidden reasoning + answer. Released ~Apr 7, 2026."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "glm-5",
        context_window: Some(200_000),
        max_output_tokens: Some(131_072),
        pricing: Some(Pricing::flat(1.0, 3.2).with_notes("Native z.ai pricing: input $1.0/M, output $3.2/M (higher than the OpenRouter GLM-5 rate of $0.60/$1.92).")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: ZAI_GLM5_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Native z.ai constructor model string `glm-5`. Reasoning/thinking model; 200K context, 128K (131072) max output per docs.z.ai/guides/llm/glm-5. Native pricing used for the native route. Released ~Feb 11, 2026."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "moonshotai/kimi-k2.6",
        context_window: Some(262_144),
        max_output_tokens: Some(65_536),
        pricing: Some(Pricing::flat(0.684, 3.42).with_notes("OpenRouter rate for moonshotai/kimi-k2.6: input $0.684/M, output $3.42/M.")),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENROUTER_KIMI_K26_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Exact OpenRouter slug (note the dot). Hybrid model marketed/used as a non-reasoning coding+multimodal model, so supports_thinking=false (use moonshotai/kimi-k2-thinking for the dedicated reasoning model). Context 262,144; 65536 is a generous app-side completion budget within the window."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "moonshotai/kimi-k2.5",
        context_window: Some(262_144),
        max_output_tokens: Some(32_768),
        pricing: Some(Pricing::flat(0.4, 1.9).with_notes("OpenRouter rate for moonshotai/kimi-k2.5: input $0.40/M, output $1.90/M.")),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENROUTER_KIMI_K25_URL,
        source_status: SourceStatus::Derived,
        notes: Some("OpenRouter route for the model the native constructor names 'kimi-k2.5'. Treated as non-reasoning (visual-coding + agentic tool-calling) on OpenRouter. Context 262,144; 32768 is a generous app-side completion budget within the window."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "kimi-k2.5",
        context_window: Some(262_144),
        max_output_tokens: Some(32_768),
        pricing: Some(Pricing::flat(0.6, 3.0).with_notes("Native Moonshot estimate from Artificial Analysis (~$0.58 in / $3.00 out); input rounded up to $0.60 to stay conservative for budget reservation.")),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: KIMI_K25_AA_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Exact native model_id used by the native constructor (Moonshot platform.kimi.ai base_url). Native pricing not on the first-party table (only k2.6 is enumerated); figures derived from Artificial Analysis. Context 262,144; 32768 is a generous within-window completion budget."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "kimi-k2-thinking",
        context_window: Some(262_144),
        max_output_tokens: Some(131_072),
        pricing: Some(Pricing::flat(0.6, 2.5).with_notes("Cross-provider median for kimi-k2-thinking (OpenRouter/Artificial Analysis): input $0.60/M, output $2.50/M, used as a conservative native estimate.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENROUTER_KIMI_K2_THINKING_URL,
        source_status: SourceStatus::Unverified,
        notes: Some("Exact native model_id used by the native constructor; a REASONING model (emits hidden chain-of-thought before the answer). Native Moonshot base_url. First-party pricing could not be isolated; figures are the cross-provider median. Context 262,144; max_output 131072 sized generously for reasoning tokens, within the window."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "deepseek/deepseek-v4-pro",
        context_window: Some(1_048_576),
        max_output_tokens: Some(384_000),
        pricing: Some(Pricing::flat(0.44, 0.87).with_notes("OpenRouter effective post-promo rate ($0.435 in rounded up to $0.44 / $0.87 out). Pre-promo regular rate was $1.74/$3.48.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENROUTER_DEEPSEEK_V4_PRO_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Primary model named in forge config; exact OpenRouter slug. Large MoE (1.6T total / 49B active), released 2026-04-24. Reasoning/thinking model; DeepSeek returns the answer in `content` and chain-of-thought in a separate `reasoning_content` field, which must be echoed back in subsequent thinking-mode turns or the API returns 400. Max output 384K (DeepSeek ceiling), sized generously for reasoning."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "deepseek-v4-pro",
        context_window: Some(1_048_576),
        max_output_tokens: Some(384_000),
        pricing: Some(Pricing::flat_with_cached(0.44, 0.87, 0.003_625).with_notes("Official DeepSeek pricing: input cache-MISS $0.435/M (rounded up to $0.44), cache-HIT $0.003625/M, output $0.87/M.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: DEEPSEEK_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Native DeepSeek API model id 'deepseek-v4-pro' (no vendor prefix). 1M context, 384K max output. Reasoning/thinking model; separate `reasoning_content` that must be echoed back in multi-turn thinking-mode requests or you get a 400. Legacy ids deepseek-reasoner/deepseek-chat now map to V4-FLASH, not Pro."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "deepseek/deepseek-v4-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(384_000),
        pricing: Some(Pricing::flat(0.15, 0.28).with_notes("DeepSeek list rate rounded up ($0.14 in -> $0.15 / $0.28 out) used instead of OpenRouter's lower fluctuating effective rate so consumers never under-reserve budget.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENROUTER_DEEPSEEK_V4_FLASH_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Sibling V4 model (cheaper routing target). Efficiency MoE (284B total / 13B active), released 2026-04-24. Reasoning/thinking model with the same reasoning_content split + mandatory pass-back-or-400 behavior as V4 Pro. Max output 384K per DeepSeek docs."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "deepseek-v4-flash",
        context_window: Some(1_048_576),
        max_output_tokens: Some(384_000),
        pricing: Some(Pricing::flat_with_cached(0.14, 0.28, 0.002_8).with_notes("Official DeepSeek pricing: input cache-MISS $0.14/M, cache-HIT $0.0028/M, output $0.28/M.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: DEEPSEEK_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Native DeepSeek API model id 'deepseek-v4-flash'. 1M context, 384K max output. Reasoning/thinking model; same content/reasoning_content split and mandatory pass-back in thinking mode. Legacy aliases deepseek-chat/deepseek-reasoner now resolve to this Flash model."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "MiniMax-M2.5",
        context_window: Some(204_800),
        max_output_tokens: Some(131_072),
        pricing: Some(Pricing::flat_with_cached(0.3, 1.2, 0.03).with_notes("Native MiniMax first-party pricing: input $0.30/M, output $1.20/M, cache-read input $0.03/M (platform.minimax.io PAYG).")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: MINIMAX_PRICING_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Native agent-sdk constructor model string 'MiniMax-M2.5' (api.minimax.io, OpenAI-compatible). Reasoning/thinking model; emits chain-of-thought in <think>...</think> tags and supports interleaved thinking. Context 204,800; max_output 131072 sized generously for hidden reasoning + answer within the window."),
    },
    ModelCapabilities {
        provider: "openai",
        model_id: "minimax/minimax-m2.5",
        context_window: Some(204_800),
        max_output_tokens: Some(131_072),
        pricing: Some(Pricing::flat(0.15, 1.15).with_notes("OpenRouter rate for minimax/minimax-m2.5: input $0.15/M, output $1.15/M (lower than MiniMax's $0.30/$1.20 first-party rate; OpenRouter prices can fluctuate, so reserve conservatively).")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: OPENROUTER_MINIMAX_M25_URL,
        source_status: SourceStatus::Derived,
        notes: Some("OpenRouter slug 'minimax/minimax-m2.5' (same M2.5 weights as native). Reasoning/thinking model. Context 204,800; max_output 131072 sized generously for hidden reasoning tokens before the answer."),
    },
];

#[must_use]
pub fn get_model_capabilities(
    provider: &str,
    model_id: &str,
) -> Option<&'static ModelCapabilities> {
    MODEL_CAPABILITIES.iter().find(|caps| {
        caps.provider.eq_ignore_ascii_case(provider) && caps.model_id.eq_ignore_ascii_case(model_id)
    })
}

#[must_use]
pub fn default_max_output_tokens(provider: &str, model_id: &str) -> Option<u32> {
    get_model_capabilities(provider, model_id).and_then(|caps| caps.max_output_tokens)
}

#[must_use]
pub const fn supported_model_capabilities() -> &'static [ModelCapabilities] {
    MODEL_CAPABILITIES
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_lookup_anthropic_sonnet_5() {
        let caps = get_model_capabilities("anthropic", "claude-sonnet-5").unwrap();
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_thinking);
        // Adaptive-only (like Opus 4.8): manual budget_tokens 400s; adaptive is required.
        assert!(caps.supports_adaptive_thinking);
    }

    #[test]
    fn test_lookup_anthropic_sonnet_46() {
        let caps = get_model_capabilities("anthropic", "claude-sonnet-4-6").unwrap();
        assert_eq!(caps.context_window, Some(1_000_000));
        assert_eq!(caps.max_output_tokens, Some(64_000));
        assert!(caps.supports_adaptive_thinking);
    }

    #[test]
    fn test_lookup_anthropic_sonnet_45_disables_adaptive_thinking() {
        let caps = get_model_capabilities("anthropic", "claude-sonnet-4-5-20250929").unwrap();
        assert!(!caps.supports_adaptive_thinking);
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
    fn test_lookup_openai_gpt56_family() -> anyhow::Result<()> {
        use anyhow::Context as _;

        for (model_id, input, cached_input, output, cache_write_note) in [
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
    fn test_lookup_openai_gpt53_codex() {
        let caps = get_model_capabilities("openai", "gpt-5.3-codex").unwrap();
        assert_eq!(caps.context_window, Some(400_000));
        assert_eq!(caps.max_output_tokens, Some(128_000));
        assert!(caps.supports_adaptive_thinking);
        assert!(caps.supports_thinking);
        assert_eq!(caps.source_status, SourceStatus::Official);
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
    fn test_lookup_open_non_reasoning_kimi_models() {
        // Kimi K2.6 / K2.5 are registered as non-reasoning coding models.
        let k26 = get_model_capabilities("openai", "moonshotai/kimi-k2.6").unwrap();
        assert!(!k26.supports_thinking);
        assert_eq!(k26.max_output_tokens, Some(65_536));
        assert!(k26.pricing.unwrap().input.unwrap().usd_per_million_tokens > 0.0);

        let k25_native = get_model_capabilities("openai", "kimi-k2.5").unwrap();
        assert!(!k25_native.supports_thinking);
        assert_eq!(k25_native.max_output_tokens, Some(32_768));
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
                caps.max_output_tokens.is_some_and(|m| m > 0),
                "max_output_tokens should be non-zero for {model_id}"
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
                input_tokens: 2_000,
                output_tokens: 1_000,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
            .unwrap();
        assert!((cost - 0.0075).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_cost_usd_with_cached_input() {
        let caps = get_model_capabilities("openai", "gpt-5.4").unwrap();
        let cost = caps
            .estimate_cost_usd(&Usage {
                input_tokens: 2_000,
                output_tokens: 1_000,
                cached_input_tokens: 1_000,
                cache_creation_input_tokens: 0,
            })
            .unwrap();
        assert!((cost - 0.01775).abs() < f64::EPSILON);
    }
}
