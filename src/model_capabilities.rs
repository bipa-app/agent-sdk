use crate::llm::Usage;

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
        let input = self.input.map(|p| p.estimate_cost_usd(usage.input_tokens));
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
const GOOGLE_MODELS_URL: &str = "https://ai.google.dev/gemini-api/docs/models";
const GOOGLE_PRICING_URL: &str = "https://ai.google.dev/gemini-api/docs/pricing";

const MODEL_CAPABILITIES: &[ModelCapabilities] = &[
    // Anthropic
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-opus-4-6",
        context_window: Some(200_000),
        max_output_tokens: Some(128_000),
        pricing: Some(Pricing::flat(15.0, 75.0).with_notes("Anthropic Opus tier pricing; verify exact current SKU mapping before billing-critical use.")),
        supports_thinking: true,
        supports_adaptive_thinking: true,
        source_url: ANTHROPIC_MODELS_URL,
        source_status: SourceStatus::Derived,
        notes: Some("Current Anthropic docs show this model alongside 200K/128K markers."),
    },
    ModelCapabilities {
        provider: "anthropic",
        model_id: "claude-sonnet-4-6",
        context_window: Some(200_000),
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
        supports_adaptive_thinking: true,
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
        supports_adaptive_thinking: true,
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
        supports_adaptive_thinking: true,
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
        supports_adaptive_thinking: true,
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
        pricing: Some(Pricing::flat(10.50, 84.0)),
        supports_thinking: false,
        supports_adaptive_thinking: false,
        source_url: OPENAI_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("Pricing verified from OpenAI pricing page. Context/max output still need clean extraction from models docs."),
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
        model_id: "gemini-3.1-pro",
        context_window: Some(1_048_576),
        max_output_tokens: Some(65_536),
        pricing: Some(Pricing::flat(2.0, 12.0).with_notes("Official pricing for prompts <= 200K tokens. For prompts > 200K, pricing increases to $4 input / $18 output per 1M tokens.")),
        supports_thinking: true,
        supports_adaptive_thinking: false,
        source_url: GOOGLE_PRICING_URL,
        source_status: SourceStatus::Official,
        notes: Some("SDK model ID omits the preview suffix; pricing sourced from Gemini 3.1 Pro Preview docs."),
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
        model_id: "gemini-3.0-flash",
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
    fn test_lookup_anthropic_sonnet_46() {
        let caps = get_model_capabilities("anthropic", "claude-sonnet-4-6").unwrap();
        assert_eq!(caps.context_window, Some(200_000));
        assert_eq!(caps.max_output_tokens, Some(64_000));
        assert!(caps.supports_adaptive_thinking);
    }

    #[test]
    fn test_lookup_openai_pricing() {
        let caps = get_model_capabilities("openai", "gpt-4o").unwrap();
        let pricing = caps.pricing.unwrap();
        assert!((pricing.input.unwrap().usd_per_million_tokens - 1.25).abs() < f64::EPSILON);
        assert!((pricing.output.unwrap().usd_per_million_tokens - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_cost_usd() {
        let caps = get_model_capabilities("openai", "gpt-4o").unwrap();
        let cost = caps
            .estimate_cost_usd(&Usage {
                input_tokens: 2_000,
                output_tokens: 1_000,
            })
            .unwrap();
        assert!((cost - 0.0075).abs() < f64::EPSILON);
    }
}
