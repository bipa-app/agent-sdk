//! Exact `OpenAI` request controls that do not fit the provider-neutral
//! [`ThinkingConfig`] abstraction.
//!
//! The shared abstraction predates `OpenAI`'s distinct `none`, `xhigh`, and
//! `max` efforts as well as GPT-5.6 reasoning modes and persisted reasoning
//! context. These additive types preserve those wire values without changing
//! the public foundation structs used by every provider.

use agent_sdk_foundation::llm::{Effort, ThinkingConfig, ThinkingMode};
use anyhow::{Result, bail};
use serde::Serialize;

/// Exact reasoning effort accepted by `OpenAI` APIs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAIReasoningEffort {
    /// Disable reasoning for latency-sensitive work.
    None,
    /// Use the smallest non-zero reasoning budget on models that support it.
    Minimal,
    /// Favor latency and token efficiency while retaining reasoning.
    Low,
    /// Balance reasoning quality, latency, and token usage.
    #[default]
    Medium,
    /// Spend more effort on difficult reasoning tasks.
    High,
    /// Use extra-high reasoning effort.
    XHigh,
    /// Use GPT-5.6's maximum reasoning effort.
    Max,
}

/// GPT-5.6 reasoning execution mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAIReasoningMode {
    /// Standard execution mode.
    #[default]
    Standard,
    /// Quality-first mode that performs additional model work.
    Pro,
}

/// Which persisted reasoning items a Responses request may reuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OpenAIReasoningContext {
    /// Use the selected model's default context policy.
    #[default]
    Auto,
    /// Make only reasoning from the current turn available.
    CurrentTurn,
    /// Reuse compatible reasoning items from earlier turns.
    AllTurns,
}

/// Requested detail level for the model's user-visible reasoning summary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAIReasoningSummary {
    /// Use the most detailed summary supported by the selected model.
    #[default]
    Auto,
    /// Request a concise reasoning summary.
    Concise,
    /// Request a detailed reasoning summary.
    Detailed,
}

/// Desired verbosity of the final text response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAITextVerbosity {
    /// Prefer concise final text.
    Low,
    /// Use the model's balanced verbosity.
    #[default]
    Medium,
    /// Prefer a more detailed final response.
    High,
}

/// Select the `OpenAI` API surface used by [`OpenAIProvider`](super::openai::OpenAIProvider).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpenAIApiSurface {
    /// Let the provider choose the best surface for the model and request.
    #[default]
    Auto,
    /// Prefer Chat Completions when the selected controls are supported there.
    ChatCompletions,
    /// Force the Responses API.
    Responses,
}

/// Controls whether GPT-5.6 creates an automatic prompt-cache breakpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAIPromptCacheMode {
    /// Let `OpenAI` create an implicit cache breakpoint.
    #[default]
    Implicit,
    /// Cache only caller-provided explicit breakpoints.
    Explicit,
}

/// Exact GPT-5.6 prompt-cache retention window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum OpenAIPromptCacheTtl {
    /// Retain the cached prefix for 30 minutes.
    #[serde(rename = "30m")]
    ThirtyMinutes,
}

/// Whether an allowed-tools policy may call zero tools or must call at least one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAIAllowedToolsMode {
    /// Let the model decide whether to call one of the allowed tools.
    #[default]
    Auto,
    /// Require at least one call from the allowed subset.
    Required,
}

/// Complete `OpenAI` function-tool selection policy.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpenAIToolChoice {
    /// Do not call a function tool.
    None,
    /// Let the model choose whether and which function tools to call.
    Auto,
    /// Require one or more function calls.
    Required,
    /// Force one named function.
    Function(String),
    /// Restrict calls to a stable subset without changing the full tool list.
    AllowedTools {
        mode: OpenAIAllowedToolsMode,
        tools: Vec<String>,
    },
}

/// Exact `OpenAI` reasoning and closely related response controls.
///
/// Every field is optional. [`Self::new`] therefore preserves the model's
/// server-side defaults until a builder is called.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenAIReasoningConfig {
    effort: Option<OpenAIReasoningEffort>,
    mode: Option<OpenAIReasoningMode>,
    context: Option<OpenAIReasoningContext>,
    summary: Option<OpenAIReasoningSummary>,
    verbosity: Option<OpenAITextVerbosity>,
    api_surface: OpenAIApiSurface,
    prompt_cache_mode: Option<OpenAIPromptCacheMode>,
    prompt_cache_ttl: Option<OpenAIPromptCacheTtl>,
    store: Option<bool>,
    parallel_tool_calls: Option<bool>,
    tool_choice: Option<OpenAIToolChoice>,
    safety_identifier: Option<String>,
}

impl OpenAIReasoningConfig {
    /// Create a config that preserves every model/API default.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            effort: None,
            mode: None,
            context: None,
            summary: None,
            verbosity: None,
            api_surface: OpenAIApiSurface::Auto,
            prompt_cache_mode: None,
            prompt_cache_ttl: None,
            store: None,
            parallel_tool_calls: None,
            tool_choice: None,
            safety_identifier: None,
        }
    }

    /// Set the exact reasoning effort.
    #[must_use]
    pub const fn with_effort(mut self, effort: OpenAIReasoningEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Set the GPT-5.6 reasoning execution mode.
    #[must_use]
    pub const fn with_mode(mut self, mode: OpenAIReasoningMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Select which persisted reasoning context the Responses API may reuse.
    #[must_use]
    pub const fn with_context(mut self, context: OpenAIReasoningContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Request a user-visible reasoning summary.
    #[must_use]
    pub const fn with_summary(mut self, summary: OpenAIReasoningSummary) -> Self {
        self.summary = Some(summary);
        self
    }

    /// Set final-answer verbosity.
    #[must_use]
    pub const fn with_verbosity(mut self, verbosity: OpenAITextVerbosity) -> Self {
        self.verbosity = Some(verbosity);
        self
    }

    /// Select the API surface used by the general `OpenAI` provider.
    #[must_use]
    pub const fn with_api_surface(mut self, api_surface: OpenAIApiSurface) -> Self {
        self.api_surface = api_surface;
        self
    }

    /// Select implicit or explicit prompt-cache breakpoint creation.
    #[must_use]
    pub const fn with_prompt_cache_mode(
        mut self,
        prompt_cache_mode: OpenAIPromptCacheMode,
    ) -> Self {
        self.prompt_cache_mode = Some(prompt_cache_mode);
        self
    }

    /// Set GPT-5.6's exact prompt-cache retention window.
    #[must_use]
    pub const fn with_prompt_cache_ttl(mut self, prompt_cache_ttl: OpenAIPromptCacheTtl) -> Self {
        self.prompt_cache_ttl = Some(prompt_cache_ttl);
        self
    }

    /// Select whether `OpenAI` stores the response as application state.
    #[must_use]
    pub const fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Enable or disable parallel function calls.
    #[must_use]
    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Set the complete `OpenAI` function-tool choice policy.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: OpenAIToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set a stable, privacy-preserving identifier for end-user safety controls.
    #[must_use]
    pub fn with_safety_identifier(mut self, safety_identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(safety_identifier.into());
        self
    }

    /// Configured exact reasoning effort, if any.
    #[must_use]
    pub const fn effort(&self) -> Option<OpenAIReasoningEffort> {
        self.effort
    }

    /// Configured reasoning execution mode, if any.
    #[must_use]
    pub const fn mode(&self) -> Option<OpenAIReasoningMode> {
        self.mode
    }

    /// Configured persisted-reasoning context policy, if any.
    #[must_use]
    pub const fn context(&self) -> Option<OpenAIReasoningContext> {
        self.context
    }

    /// Configured reasoning summary detail, if any.
    #[must_use]
    pub const fn summary(&self) -> Option<OpenAIReasoningSummary> {
        self.summary
    }

    /// Configured final-answer verbosity, if any.
    #[must_use]
    pub const fn verbosity(&self) -> Option<OpenAITextVerbosity> {
        self.verbosity
    }

    /// Selected API surface.
    #[must_use]
    pub const fn api_surface(&self) -> OpenAIApiSurface {
        self.api_surface
    }

    /// Configured prompt-cache mode, if any.
    #[must_use]
    pub const fn prompt_cache_mode(&self) -> Option<OpenAIPromptCacheMode> {
        self.prompt_cache_mode
    }

    /// Configured exact prompt-cache TTL, if any.
    #[must_use]
    pub const fn prompt_cache_ttl(&self) -> Option<OpenAIPromptCacheTtl> {
        self.prompt_cache_ttl
    }

    /// Explicit response-storage preference, if configured.
    #[must_use]
    pub const fn store(&self) -> Option<bool> {
        self.store
    }

    /// Explicit parallel-tool preference, if configured.
    #[must_use]
    pub const fn parallel_tool_calls(&self) -> Option<bool> {
        self.parallel_tool_calls
    }

    /// Provider-owned tool-choice policy, if configured.
    #[must_use]
    pub const fn tool_choice(&self) -> Option<&OpenAIToolChoice> {
        self.tool_choice.as_ref()
    }

    /// Stable safety identifier, if configured.
    #[must_use]
    pub fn safety_identifier(&self) -> Option<&str> {
        self.safety_identifier.as_deref()
    }

    /// Whether this config uses controls that only the Responses API accepts.
    #[must_use]
    pub const fn requires_responses_api(&self) -> bool {
        matches!(self.api_surface, OpenAIApiSurface::Responses)
            || self.mode.is_some()
            || self.context.is_some()
            || self.summary.is_some()
    }

    pub(crate) const fn with_optional_effort(
        mut self,
        effort: Option<OpenAIReasoningEffort>,
    ) -> Self {
        self.effort = effort;
        self
    }
}

pub(crate) const fn is_gpt56_model(model: &str) -> bool {
    matches!(
        model.as_bytes(),
        b"gpt-5.6" | b"gpt-5.6-sol" | b"gpt-5.6-terra" | b"gpt-5.6-luna"
    )
}

pub(crate) fn validate_reasoning_config(model: &str, config: &OpenAIReasoningConfig) -> Result<()> {
    if let Some(effort) = config.effort() {
        let validation = if is_gpt56_model(model) {
            Some((
                matches!(
                    effort,
                    OpenAIReasoningEffort::None
                        | OpenAIReasoningEffort::Low
                        | OpenAIReasoningEffort::Medium
                        | OpenAIReasoningEffort::High
                        | OpenAIReasoningEffort::XHigh
                        | OpenAIReasoningEffort::Max
                ),
                "none, low, medium, high, xhigh, and max",
            ))
        } else {
            match model {
                "gpt-5.4" => Some((
                    matches!(
                        effort,
                        OpenAIReasoningEffort::None
                            | OpenAIReasoningEffort::Low
                            | OpenAIReasoningEffort::Medium
                            | OpenAIReasoningEffort::High
                            | OpenAIReasoningEffort::XHigh
                    ),
                    "none, low, medium, high, and xhigh",
                )),
                "gpt-5.3-codex" => Some((
                    matches!(
                        effort,
                        OpenAIReasoningEffort::Low
                            | OpenAIReasoningEffort::Medium
                            | OpenAIReasoningEffort::High
                            | OpenAIReasoningEffort::XHigh
                    ),
                    "low, medium, high, and xhigh",
                )),
                "gpt-5.2-pro" => Some((
                    matches!(
                        effort,
                        OpenAIReasoningEffort::Medium
                            | OpenAIReasoningEffort::High
                            | OpenAIReasoningEffort::XHigh
                    ),
                    "medium, high, and xhigh",
                )),
                "gpt-5" | "gpt-5-mini" | "gpt-5-nano" => Some((
                    matches!(
                        effort,
                        OpenAIReasoningEffort::Minimal
                            | OpenAIReasoningEffort::Low
                            | OpenAIReasoningEffort::Medium
                            | OpenAIReasoningEffort::High
                    ),
                    "minimal, low, medium, and high",
                )),
                _ => None,
            }
        };

        if let Some((false, supported_names)) = validation {
            bail!(
                "reasoning effort is not supported for model={model}; supported efforts are {supported_names}"
            );
        }
    }

    if (config.prompt_cache_mode().is_some() || config.prompt_cache_ttl().is_some())
        && !is_gpt56_model(model)
    {
        bail!(
            "exact prompt-cache mode and TTL controls are only supported for GPT-5.6 models; model={model}"
        );
    }

    if model == "gpt-5.3-codex"
        && (config.mode().is_some() || config.context().is_some() || config.summary().is_some())
    {
        bail!("reasoning mode, context, and summary controls are not supported for model={model}");
    }

    Ok(())
}

pub(crate) fn validate_tool_choice(
    config: Option<&OpenAIReasoningConfig>,
    tools: Option<&[agent_sdk_foundation::llm::Tool]>,
) -> Result<()> {
    let Some(choice) = config.and_then(OpenAIReasoningConfig::tool_choice) else {
        return Ok(());
    };
    let tools = tools.unwrap_or_default();

    match choice {
        OpenAIToolChoice::Required if tools.is_empty() => {
            bail!("OpenAI tool_choice=required needs at least one function tool")
        }
        OpenAIToolChoice::Function(name) => {
            if !tools.iter().any(|tool| tool.name == *name) {
                bail!("OpenAI tool_choice names unknown function `{name}`");
            }
        }
        OpenAIToolChoice::AllowedTools { tools: allowed, .. } => {
            if allowed.is_empty() {
                bail!("OpenAI allowed_tools must contain at least one function name");
            }
            if let Some(name) = allowed
                .iter()
                .find(|name| !tools.iter().any(|tool| tool.name == name.as_str()))
            {
                bail!("OpenAI allowed_tools names unknown function `{name}`");
            }
        }
        OpenAIToolChoice::None | OpenAIToolChoice::Auto | OpenAIToolChoice::Required => {}
    }

    Ok(())
}

pub(crate) const fn legacy_reasoning_effort(
    config: &ThinkingConfig,
) -> Option<OpenAIReasoningEffort> {
    if let Some(effort) = config.effort {
        return Some(match effort {
            Effort::Low => OpenAIReasoningEffort::Low,
            Effort::Medium => OpenAIReasoningEffort::Medium,
            Effort::High => OpenAIReasoningEffort::High,
            Effort::XHigh => OpenAIReasoningEffort::XHigh,
            Effort::Max => OpenAIReasoningEffort::Max,
        });
    }

    match &config.mode {
        ThinkingMode::Adaptive | ThinkingMode::Default => None,
        ThinkingMode::Enabled { budget_tokens } => Some(if *budget_tokens <= 4_096 {
            OpenAIReasoningEffort::Low
        } else if *budget_tokens <= 16_384 {
            OpenAIReasoningEffort::Medium
        } else if *budget_tokens <= 32_768 {
            OpenAIReasoningEffort::High
        } else {
            OpenAIReasoningEffort::XHigh
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_efforts_serialize_without_collapsing_xhigh_and_max() -> anyhow::Result<()> {
        for (effort, expected) in [
            (OpenAIReasoningEffort::None, "\"none\""),
            (OpenAIReasoningEffort::Minimal, "\"minimal\""),
            (OpenAIReasoningEffort::Low, "\"low\""),
            (OpenAIReasoningEffort::Medium, "\"medium\""),
            (OpenAIReasoningEffort::High, "\"high\""),
            (OpenAIReasoningEffort::XHigh, "\"xhigh\""),
            (OpenAIReasoningEffort::Max, "\"max\""),
        ] {
            assert_eq!(serde_json::to_string(&effort)?, expected);
        }
        Ok(())
    }

    #[test]
    fn legacy_effort_maps_xhigh_and_max_exactly() {
        let xhigh = ThinkingConfig::adaptive_with_effort(Effort::XHigh);
        assert_eq!(
            legacy_reasoning_effort(&xhigh),
            Some(OpenAIReasoningEffort::XHigh)
        );
        let max = ThinkingConfig::adaptive_with_effort(Effort::Max);
        assert_eq!(
            legacy_reasoning_effort(&max),
            Some(OpenAIReasoningEffort::Max)
        );
    }

    #[test]
    fn response_only_controls_are_detected() {
        assert!(!OpenAIReasoningConfig::new().requires_responses_api());
        assert!(
            OpenAIReasoningConfig::new()
                .with_mode(OpenAIReasoningMode::Pro)
                .requires_responses_api()
        );
        assert!(
            OpenAIReasoningConfig::new()
                .with_mode(OpenAIReasoningMode::Standard)
                .requires_responses_api()
        );
        assert!(
            OpenAIReasoningConfig::new()
                .with_context(OpenAIReasoningContext::AllTurns)
                .requires_responses_api()
        );
        assert!(
            OpenAIReasoningConfig::new()
                .with_summary(OpenAIReasoningSummary::Auto)
                .requires_responses_api()
        );
        assert!(
            !OpenAIReasoningConfig::new()
                .with_tool_choice(OpenAIToolChoice::AllowedTools {
                    mode: OpenAIAllowedToolsMode::Auto,
                    tools: vec!["lookup".to_owned()],
                })
                .requires_responses_api()
        );
    }

    #[test]
    fn gpt56_rejects_minimal_but_accepts_max() {
        let minimal = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::Minimal);
        assert!(validate_reasoning_config("gpt-5.6", &minimal).is_err());

        let max = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::Max);
        assert!(validate_reasoning_config("gpt-5.6", &max).is_ok());
    }

    #[test]
    fn gpt53_codex_enforces_its_narrower_reasoning_and_cache_controls() {
        let xhigh = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::XHigh);
        assert!(validate_reasoning_config("gpt-5.3-codex", &xhigh).is_ok());

        let max = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::Max);
        assert!(validate_reasoning_config("gpt-5.3-codex", &max).is_err());

        let mode = OpenAIReasoningConfig::new().with_mode(OpenAIReasoningMode::Pro);
        assert!(validate_reasoning_config("gpt-5.3-codex", &mode).is_err());

        let cache =
            OpenAIReasoningConfig::new().with_prompt_cache_mode(OpenAIPromptCacheMode::Explicit);
        assert!(validate_reasoning_config("gpt-5.3-codex", &cache).is_err());
    }

    #[test]
    fn known_model_effort_sets_reject_unsupported_values() {
        let minimal = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::Minimal);
        let none = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::None);
        let xhigh = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::XHigh);
        let max = OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::Max);

        assert!(validate_reasoning_config("gpt-5.4", &none).is_ok());
        assert!(validate_reasoning_config("gpt-5.4", &minimal).is_err());
        assert!(validate_reasoning_config("gpt-5.4", &max).is_err());

        assert!(validate_reasoning_config("gpt-5.2-pro", &none).is_err());
        assert!(validate_reasoning_config("gpt-5.2-pro", &minimal).is_err());
        assert!(
            validate_reasoning_config(
                "gpt-5.2-pro",
                &OpenAIReasoningConfig::new().with_effort(OpenAIReasoningEffort::Medium),
            )
            .is_ok()
        );
        assert!(validate_reasoning_config("gpt-5.2-pro", &xhigh).is_ok());
        assert!(validate_reasoning_config("gpt-5.2-pro", &max).is_err());

        for model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"] {
            assert!(validate_reasoning_config(model, &minimal).is_ok());
            assert!(validate_reasoning_config(model, &none).is_err());
            assert!(validate_reasoning_config(model, &xhigh).is_err());
            assert!(validate_reasoning_config(model, &max).is_err());
        }

        let custom_max = validate_reasoning_config("vendor/custom-reasoner", &max);
        assert!(custom_max.is_ok());
    }
}
