//! Static feature metadata for `OpenAI` models.
//!
//! This registry complements [`crate::model_capabilities`] with API-shape
//! information used to validate requests and choose an inference surface.

/// `OpenAI` API surface on which a model can be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelApiSurface {
    ChatCompletions,
    Responses,
    Batch,
}

/// Input modality accepted by a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelInputModality {
    Text,
    Image,
}

/// Exact `OpenAI` `reasoning.effort` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// Exact `OpenAI` `reasoning.mode` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelReasoningMode {
    Standard,
    Pro,
}

/// Exact `OpenAI` `reasoning.context` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelReasoningContext {
    Auto,
    CurrentTurn,
    AllTurns,
}

/// Exact `OpenAI` `reasoning.summary` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

/// Mechanism available for continuing opaque reasoning state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelReasoningStateReplay {
    PreviousResponseId,
    ManualOutputItems,
    EncryptedContent,
}

/// Exact GPT-5.6 `prompt_cache_options.mode` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelPromptCacheMode {
    Implicit,
    Explicit,
}

/// Exact GPT-5.6 `prompt_cache_options.ttl` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelPromptCacheTtl {
    ThirtyMinutes,
}

/// Tool-selection policy accepted by `OpenAI` function calling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelToolChoice {
    None,
    Auto,
    Required,
    ForcedFunction,
    AllowedTools,
}

/// Values supported by a feature and the request surfaces that accept them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelFeatureSet<T: 'static> {
    pub values: &'static [T],
    pub api_surfaces: &'static [ModelApiSurface],
}

/// Reasoning controls supported by a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelReasoningFeatures {
    pub efforts: ModelFeatureSet<ModelReasoningEffort>,
    pub modes: ModelFeatureSet<ModelReasoningMode>,
    pub contexts: ModelFeatureSet<ModelReasoningContext>,
    pub summaries: ModelFeatureSet<ModelReasoningSummary>,
    pub state_replay: ModelFeatureSet<ModelReasoningStateReplay>,
}

/// Prompt-cache controls supported by a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelPromptCacheFeatures {
    pub automatic: &'static [ModelApiSurface],
    pub prompt_cache_key: &'static [ModelApiSurface],
    pub modes: ModelFeatureSet<ModelPromptCacheMode>,
    pub ttls: ModelFeatureSet<ModelPromptCacheTtl>,
    pub explicit_breakpoints: &'static [ModelApiSurface],
    pub max_write_breakpoints: Option<u8>,
}

/// Function-tool controls supported by a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelToolFeatures {
    pub choices: ModelFeatureSet<ModelToolChoice>,
    pub parallel_function_calls: &'static [ModelApiSurface],
}

/// API feature profile for one exact `OpenAI` model ID or alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelFeatures {
    /// Exact request model ID.
    pub model_id: &'static str,
    /// Canonical model selected by this alias, when applicable.
    pub alias_of: Option<&'static str>,
    /// Total input-plus-output context capacity.
    pub context_window: u32,
    /// Maximum input tokens accepted in one request.
    pub max_input_tokens: u32,
    /// Maximum output-token budget accepted in one request.
    pub max_output_tokens: u32,
    pub api_surfaces: &'static [ModelApiSurface],
    pub input_modalities: &'static [ModelInputModality],
    pub reasoning: ModelReasoningFeatures,
    pub prompt_cache: ModelPromptCacheFeatures,
    pub tools: ModelToolFeatures,
    pub source_urls: &'static [&'static str],
}

const CHAT_AND_RESPONSES: &[ModelApiSurface] =
    &[ModelApiSurface::ChatCompletions, ModelApiSurface::Responses];
const RESPONSES: &[ModelApiSurface] = &[ModelApiSurface::Responses];
const ALL_SURFACES: &[ModelApiSurface] = &[
    ModelApiSurface::ChatCompletions,
    ModelApiSurface::Responses,
    ModelApiSurface::Batch,
];
const TEXT_AND_IMAGE: &[ModelInputModality] =
    &[ModelInputModality::Text, ModelInputModality::Image];

const GPT56_EFFORTS: &[ModelReasoningEffort] = &[
    ModelReasoningEffort::None,
    ModelReasoningEffort::Low,
    ModelReasoningEffort::Medium,
    ModelReasoningEffort::High,
    ModelReasoningEffort::XHigh,
    ModelReasoningEffort::Max,
];
const GPT53_CODEX_EFFORTS: &[ModelReasoningEffort] = &[
    ModelReasoningEffort::Low,
    ModelReasoningEffort::Medium,
    ModelReasoningEffort::High,
    ModelReasoningEffort::XHigh,
];
const GPT52_PRO_EFFORTS: &[ModelReasoningEffort] = &[
    ModelReasoningEffort::Medium,
    ModelReasoningEffort::High,
    ModelReasoningEffort::XHigh,
];
const GPT56_MODES: &[ModelReasoningMode] = &[ModelReasoningMode::Standard, ModelReasoningMode::Pro];
const GPT56_CONTEXTS: &[ModelReasoningContext] = &[
    ModelReasoningContext::Auto,
    ModelReasoningContext::CurrentTurn,
    ModelReasoningContext::AllTurns,
];
const AUTO_SUMMARY: &[ModelReasoningSummary] = &[ModelReasoningSummary::Auto];
const REASONING_STATE_REPLAY: &[ModelReasoningStateReplay] = &[
    ModelReasoningStateReplay::PreviousResponseId,
    ModelReasoningStateReplay::ManualOutputItems,
    ModelReasoningStateReplay::EncryptedContent,
];
const CACHE_MODES: &[ModelPromptCacheMode] = &[
    ModelPromptCacheMode::Implicit,
    ModelPromptCacheMode::Explicit,
];
const CACHE_TTLS: &[ModelPromptCacheTtl] = &[ModelPromptCacheTtl::ThirtyMinutes];
const TOOL_CHOICES: &[ModelToolChoice] = &[
    ModelToolChoice::None,
    ModelToolChoice::Auto,
    ModelToolChoice::Required,
    ModelToolChoice::ForcedFunction,
    ModelToolChoice::AllowedTools,
];

const GPT56_REASONING: ModelReasoningFeatures = ModelReasoningFeatures {
    efforts: ModelFeatureSet {
        values: GPT56_EFFORTS,
        api_surfaces: CHAT_AND_RESPONSES,
    },
    modes: ModelFeatureSet {
        values: GPT56_MODES,
        api_surfaces: RESPONSES,
    },
    contexts: ModelFeatureSet {
        values: GPT56_CONTEXTS,
        api_surfaces: RESPONSES,
    },
    summaries: ModelFeatureSet {
        values: AUTO_SUMMARY,
        api_surfaces: RESPONSES,
    },
    state_replay: ModelFeatureSet {
        values: REASONING_STATE_REPLAY,
        api_surfaces: RESPONSES,
    },
};

const GPT53_CODEX_REASONING: ModelReasoningFeatures = ModelReasoningFeatures {
    efforts: ModelFeatureSet {
        values: GPT53_CODEX_EFFORTS,
        api_surfaces: RESPONSES,
    },
    modes: ModelFeatureSet {
        values: &[],
        api_surfaces: &[],
    },
    contexts: ModelFeatureSet {
        values: &[],
        api_surfaces: &[],
    },
    summaries: ModelFeatureSet {
        values: &[],
        api_surfaces: &[],
    },
    state_replay: ModelFeatureSet {
        values: REASONING_STATE_REPLAY,
        api_surfaces: RESPONSES,
    },
};

const GPT52_PRO_REASONING: ModelReasoningFeatures = ModelReasoningFeatures {
    efforts: ModelFeatureSet {
        values: GPT52_PRO_EFFORTS,
        api_surfaces: RESPONSES,
    },
    modes: ModelFeatureSet {
        values: &[],
        api_surfaces: &[],
    },
    contexts: ModelFeatureSet {
        values: &[],
        api_surfaces: &[],
    },
    summaries: ModelFeatureSet {
        values: &[],
        api_surfaces: &[],
    },
    state_replay: ModelFeatureSet {
        values: REASONING_STATE_REPLAY,
        api_surfaces: RESPONSES,
    },
};

const GPT56_CACHE: ModelPromptCacheFeatures = ModelPromptCacheFeatures {
    automatic: CHAT_AND_RESPONSES,
    prompt_cache_key: CHAT_AND_RESPONSES,
    modes: ModelFeatureSet {
        values: CACHE_MODES,
        api_surfaces: CHAT_AND_RESPONSES,
    },
    ttls: ModelFeatureSet {
        values: CACHE_TTLS,
        api_surfaces: CHAT_AND_RESPONSES,
    },
    explicit_breakpoints: CHAT_AND_RESPONSES,
    max_write_breakpoints: Some(4),
};

const LEGACY_AUTOMATIC_CACHE: ModelPromptCacheFeatures = ModelPromptCacheFeatures {
    automatic: RESPONSES,
    prompt_cache_key: RESPONSES,
    modes: ModelFeatureSet {
        values: &[],
        api_surfaces: &[],
    },
    ttls: ModelFeatureSet {
        values: &[],
        api_surfaces: &[],
    },
    explicit_breakpoints: &[],
    max_write_breakpoints: None,
};

const FUNCTION_TOOLS: ModelToolFeatures = ModelToolFeatures {
    choices: ModelFeatureSet {
        values: TOOL_CHOICES,
        api_surfaces: CHAT_AND_RESPONSES,
    },
    parallel_function_calls: CHAT_AND_RESPONSES,
};

const RESPONSES_FUNCTION_TOOLS: ModelToolFeatures = ModelToolFeatures {
    choices: ModelFeatureSet {
        values: TOOL_CHOICES,
        api_surfaces: RESPONSES,
    },
    parallel_function_calls: RESPONSES,
};

const REASONING_URL: &str = "https://developers.openai.com/api/docs/guides/reasoning";
const CACHE_URL: &str = "https://developers.openai.com/api/docs/guides/prompt-caching";
const FUNCTION_CALLING_URL: &str = "https://developers.openai.com/api/docs/guides/function-calling";
const GPT56_GUIDE_URL: &str = "https://developers.openai.com/api/docs/guides/latest-model";
const GPT56_SOL_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-sol";
const GPT56_TERRA_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-terra";
const GPT56_LUNA_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.6-luna";
const GPT53_CODEX_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.3-codex";
const GPT52_PRO_URL: &str = "https://developers.openai.com/api/docs/models/gpt-5.2-pro";

const GPT56_SOURCES: &[&str] = &[
    GPT56_SOL_URL,
    GPT56_GUIDE_URL,
    REASONING_URL,
    CACHE_URL,
    FUNCTION_CALLING_URL,
];
const GPT56_SOL_SOURCES: &[&str] = GPT56_SOURCES;
const GPT56_TERRA_SOURCES: &[&str] = &[
    GPT56_TERRA_URL,
    GPT56_GUIDE_URL,
    REASONING_URL,
    CACHE_URL,
    FUNCTION_CALLING_URL,
];
const GPT56_LUNA_SOURCES: &[&str] = &[
    GPT56_LUNA_URL,
    GPT56_GUIDE_URL,
    REASONING_URL,
    CACHE_URL,
    FUNCTION_CALLING_URL,
];
const GPT53_CODEX_SOURCES: &[&str] = &[
    GPT53_CODEX_URL,
    REASONING_URL,
    CACHE_URL,
    FUNCTION_CALLING_URL,
];
const GPT52_PRO_SOURCES: &[&str] = &[
    GPT52_PRO_URL,
    REASONING_URL,
    CACHE_URL,
    FUNCTION_CALLING_URL,
];

const MODEL_FEATURES: &[ModelFeatures] = &[
    ModelFeatures {
        model_id: "gpt-5.6",
        alias_of: Some("gpt-5.6-sol"),
        context_window: 1_050_000,
        max_input_tokens: 922_000,
        max_output_tokens: 128_000,
        api_surfaces: ALL_SURFACES,
        input_modalities: TEXT_AND_IMAGE,
        reasoning: GPT56_REASONING,
        prompt_cache: GPT56_CACHE,
        tools: FUNCTION_TOOLS,
        source_urls: GPT56_SOURCES,
    },
    ModelFeatures {
        model_id: "gpt-5.6-sol",
        alias_of: None,
        context_window: 1_050_000,
        max_input_tokens: 922_000,
        max_output_tokens: 128_000,
        api_surfaces: ALL_SURFACES,
        input_modalities: TEXT_AND_IMAGE,
        reasoning: GPT56_REASONING,
        prompt_cache: GPT56_CACHE,
        tools: FUNCTION_TOOLS,
        source_urls: GPT56_SOL_SOURCES,
    },
    ModelFeatures {
        model_id: "gpt-5.6-terra",
        alias_of: None,
        context_window: 1_050_000,
        max_input_tokens: 922_000,
        max_output_tokens: 128_000,
        api_surfaces: ALL_SURFACES,
        input_modalities: TEXT_AND_IMAGE,
        reasoning: GPT56_REASONING,
        prompt_cache: GPT56_CACHE,
        tools: FUNCTION_TOOLS,
        source_urls: GPT56_TERRA_SOURCES,
    },
    ModelFeatures {
        model_id: "gpt-5.6-luna",
        alias_of: None,
        context_window: 1_050_000,
        max_input_tokens: 922_000,
        max_output_tokens: 128_000,
        api_surfaces: ALL_SURFACES,
        input_modalities: TEXT_AND_IMAGE,
        reasoning: GPT56_REASONING,
        prompt_cache: GPT56_CACHE,
        tools: FUNCTION_TOOLS,
        source_urls: GPT56_LUNA_SOURCES,
    },
    ModelFeatures {
        model_id: "gpt-5.3-codex",
        alias_of: None,
        context_window: 400_000,
        max_input_tokens: 272_000,
        max_output_tokens: 128_000,
        api_surfaces: RESPONSES,
        input_modalities: TEXT_AND_IMAGE,
        reasoning: GPT53_CODEX_REASONING,
        prompt_cache: LEGACY_AUTOMATIC_CACHE,
        tools: RESPONSES_FUNCTION_TOOLS,
        source_urls: GPT53_CODEX_SOURCES,
    },
    ModelFeatures {
        model_id: "gpt-5.2-pro",
        alias_of: None,
        context_window: 400_000,
        max_input_tokens: 272_000,
        max_output_tokens: 128_000,
        api_surfaces: RESPONSES,
        input_modalities: TEXT_AND_IMAGE,
        reasoning: GPT52_PRO_REASONING,
        prompt_cache: LEGACY_AUTOMATIC_CACHE,
        tools: RESPONSES_FUNCTION_TOOLS,
        source_urls: GPT52_PRO_SOURCES,
    },
];

/// Return the static feature profile for an exact model ID or alias.
#[must_use]
pub fn get_model_features(model_id: &str) -> Option<&'static ModelFeatures> {
    MODEL_FEATURES
        .iter()
        .find(|features| features.model_id == model_id)
}

/// Return all model feature profiles bundled with this SDK release.
#[must_use]
pub const fn supported_model_features() -> &'static [ModelFeatures] {
    MODEL_FEATURES
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn registry_contains_every_official_row() -> anyhow::Result<()> {
        for model_id in [
            "gpt-5.6",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.3-codex",
            "gpt-5.2-pro",
        ] {
            get_model_features(model_id)
                .with_context(|| format!("missing model feature row for {model_id}"))?;
        }
        assert!(get_model_features("unknown-model").is_none());
        Ok(())
    }

    #[test]
    fn gpt56_alias_preserves_exact_surface_and_reasoning_features() -> anyhow::Result<()> {
        let features = get_model_features("gpt-5.6").context("missing gpt-5.6 features")?;

        assert_eq!(features.alias_of, Some("gpt-5.6-sol"));
        assert_eq!(features.context_window, 1_050_000);
        assert_eq!(features.max_input_tokens, 922_000);
        assert_eq!(features.max_output_tokens, 128_000);
        assert_eq!(features.api_surfaces, ALL_SURFACES);
        assert_eq!(features.input_modalities, TEXT_AND_IMAGE);
        assert_eq!(features.reasoning.efforts.values, GPT56_EFFORTS);
        assert_eq!(features.reasoning.modes.values, GPT56_MODES);
        assert_eq!(features.reasoning.contexts.values, GPT56_CONTEXTS);
        assert_eq!(features.reasoning.summaries.values, AUTO_SUMMARY);
        assert_eq!(
            features.reasoning.state_replay.values,
            REASONING_STATE_REPLAY
        );
        Ok(())
    }

    #[test]
    fn gpt56_exposes_explicit_cache_and_parallel_tool_controls() -> anyhow::Result<()> {
        let features =
            get_model_features("gpt-5.6-terra").context("missing gpt-5.6-terra features")?;

        assert_eq!(features.prompt_cache.modes.values, CACHE_MODES);
        assert_eq!(features.prompt_cache.ttls.values, CACHE_TTLS);
        assert_eq!(features.prompt_cache.max_write_breakpoints, Some(4));
        assert_eq!(features.tools.choices.values, TOOL_CHOICES);
        assert_eq!(features.tools.parallel_function_calls, CHAT_AND_RESPONSES);
        Ok(())
    }

    #[test]
    fn gpt53_codex_keeps_legacy_boundaries() -> anyhow::Result<()> {
        let features =
            get_model_features("gpt-5.3-codex").context("missing gpt-5.3-codex features")?;

        assert_eq!(features.reasoning.efforts.values, GPT53_CODEX_EFFORTS);
        assert_eq!(features.api_surfaces, RESPONSES);
        assert_eq!(features.context_window, 400_000);
        assert_eq!(features.max_input_tokens, 272_000);
        assert_eq!(features.max_output_tokens, 128_000);
        assert_eq!(features.reasoning.efforts.api_surfaces, RESPONSES);
        assert!(features.reasoning.modes.values.is_empty());
        assert!(features.reasoning.contexts.values.is_empty());
        assert!(features.reasoning.summaries.values.is_empty());
        assert!(features.prompt_cache.modes.values.is_empty());
        assert!(features.prompt_cache.explicit_breakpoints.is_empty());
        assert_eq!(
            features.reasoning.state_replay.values,
            REASONING_STATE_REPLAY
        );
        Ok(())
    }

    #[test]
    fn gpt52_pro_is_responses_only_with_verified_efforts() -> anyhow::Result<()> {
        let features = get_model_features("gpt-5.2-pro").context("missing gpt-5.2-pro features")?;

        assert_eq!(features.api_surfaces, RESPONSES);
        assert_eq!(features.reasoning.efforts.values, GPT52_PRO_EFFORTS);
        assert_eq!(features.reasoning.efforts.api_surfaces, RESPONSES);
        assert_eq!(features.context_window, 400_000);
        assert_eq!(features.max_output_tokens, 128_000);
        Ok(())
    }

    #[test]
    fn model_ids_are_unique() {
        for (index, features) in supported_model_features().iter().enumerate() {
            assert!(
                supported_model_features()[index + 1..]
                    .iter()
                    .all(|candidate| candidate.model_id != features.model_id)
            );
        }
    }
}
