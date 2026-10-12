//! Configuration for context compaction.

use serde::{Deserialize, Serialize};

/// Default cap on the estimated tokens retained in the post-compaction tail.
///
/// This bounds `retain_recent`: even when `retain_recent` asks to keep many
/// messages, the retained tail is trimmed so its estimated token count stays
/// at or near this value (it is a soft cap — an indivisible tool pair may push
/// the tail slightly over).
pub const DEFAULT_MAX_RETAINED_TAIL_TOKENS: usize = 20_000;

/// Default `max_tokens` budget for the LLM summarization call.
///
/// The summarization prompt asks for 500-1000 words, which can exceed the old
/// hardcoded 2000-token ceiling for dense technical content. This default
/// leaves headroom; truncation (a `MaxTokens` stop reason) is still detected
/// and retried.
pub const DEFAULT_SUMMARY_MAX_TOKENS: usize = 4_096;

/// Fallback reserve for the model-aware compaction threshold.
///
/// Used when neither an explicit reserve nor the model's registered
/// `max_output_tokens` is known; the reserve never drops below the
/// `max_tokens` a call requests.
pub const DEFAULT_MODEL_CONTEXT_RESERVE_TOKENS: usize = 16_384;

const fn default_max_retained_tail_tokens() -> usize {
    DEFAULT_MAX_RETAINED_TAIL_TOKENS
}

const fn default_summary_max_tokens() -> usize {
    DEFAULT_SUMMARY_MAX_TOKENS
}
const fn default_compaction_engine() -> CompactionEngine {
    CompactionEngine::Legacy
}

pub use agent_sdk_foundation::events::CompactionEngine;

/// Configuration for context compaction.
///
/// Controls when and how context compaction occurs.
///
/// # Example
///
/// ```
/// use agent_sdk::context::CompactionConfig;
///
/// let config = CompactionConfig::default()
///     .with_threshold_tokens(100_000)
///     .with_retain_recent(20);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Token threshold to trigger compaction.
    /// When estimated tokens exceed this, compaction is triggered.
    /// Default: 80,000 (conservative for 128K context models)
    pub threshold_tokens: usize,

    /// Number of recent messages to keep intact (not summarized).
    /// These messages remain in full to preserve immediate context.
    ///
    /// The retained tail is additionally bounded by
    /// [`max_retained_tail_tokens`](Self::max_retained_tail_tokens): if keeping
    /// `retain_recent` messages would exceed that token budget, the oldest of
    /// those messages are summarized instead so the tail stays within the cap.
    /// Default: 10
    pub retain_recent: usize,

    /// Minimum messages before compaction is considered.
    /// Prevents compaction when conversation is still short.
    /// Default: 20
    pub min_messages_for_compaction: usize,

    /// Whether to automatically compact when threshold is reached.
    /// If false, compaction only occurs on explicit request.
    /// Default: true
    pub auto_compact: bool,

    /// Soft cap on the estimated tokens kept in the retained tail.
    ///
    /// Bounds [`retain_recent`](Self::retain_recent): the most recent messages
    /// are kept only until this token budget is reached, after which older
    /// messages are folded into the summary instead. The cap is soft because an
    /// indivisible `tool_use`/`tool_result` pair may push the tail slightly
    /// over. Raise it on large-context models, or lower it for more aggressive
    /// compaction.
    /// Default: 20,000
    #[serde(default = "default_max_retained_tail_tokens")]
    pub max_retained_tail_tokens: usize,

    /// `max_tokens` budget for the LLM summarization call.
    ///
    /// If the summarizer hits this ceiling (a `MaxTokens` stop reason), the
    /// compactor logs a warning and retries once with a larger budget before
    /// marking the summary as truncated.
    /// Default: 4,096
    #[serde(default = "default_summary_max_tokens")]
    pub summary_max_tokens: usize,

    /// Compaction implementation selected by the host.
    ///
    /// The new engine is explicit until its host completes the ENG-9528 soak.
    /// Existing hosts therefore retain legacy behavior by default.
    #[serde(default = "default_compaction_engine")]
    pub engine: CompactionEngine,

    /// Derive the trigger threshold per call from the serving model instead
    /// of treating [`threshold_tokens`](Self::threshold_tokens) as
    /// authoritative. Honored by the daemon root-turn path
    /// (`agent-server`); the in-process agent loop ignores it.
    ///
    /// The threshold is `window - reserve`. The window is
    /// [`model_context_window_tokens`](Self::model_context_window_tokens),
    /// else the serving model's registered context window. The reserve is
    /// [`model_context_reserve_tokens`](Self::model_context_reserve_tokens),
    /// else the model's registered `max_output_tokens` bounded the way the
    /// worker bounds requests (a quarter of that window, at least 32,000),
    /// else [`DEFAULT_MODEL_CONTEXT_RESERVE_TOKENS`], and never less than the
    /// `max_tokens` the call requests. Without a window (no override, and an
    /// unknown model or a provider whose route is not stable per request),
    /// `threshold_tokens` applies unchanged.
    ///
    /// While this threshold drives automatic compaction (`auto_compact` on
    /// and a compaction provider wired) and the serving model's capabilities
    /// are known, the worker also caps each request's `max_tokens` at a
    /// quarter of the same window plus the reasoning the request carries:
    /// its thinking budget, or another quarter for adaptive, effort-only or
    /// on-by-default reasoning. Otherwise only the model's registered
    /// `max_output_tokens` caps it.
    /// Default: false
    #[serde(default)]
    pub model_aware_threshold: bool,

    /// Reserve override for
    /// [`model_aware_threshold`](Self::model_aware_threshold); still raised
    /// to the call's requested `max_tokens` when smaller.
    /// Default: `None`
    #[serde(default)]
    pub model_context_reserve_tokens: Option<usize>,

    /// Context-window override for
    /// [`model_aware_threshold`](Self::model_aware_threshold).
    /// Default: `None`
    #[serde(default)]
    pub model_context_window_tokens: Option<usize>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            threshold_tokens: 80_000,
            retain_recent: 10,
            min_messages_for_compaction: 20,
            auto_compact: true,
            max_retained_tail_tokens: DEFAULT_MAX_RETAINED_TAIL_TOKENS,
            summary_max_tokens: DEFAULT_SUMMARY_MAX_TOKENS,
            engine: CompactionEngine::Legacy,
            model_aware_threshold: false,
            model_context_reserve_tokens: None,
            model_context_window_tokens: None,
        }
    }
}

impl CompactionConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the token threshold for compaction.
    #[must_use]
    pub const fn with_threshold_tokens(mut self, threshold: usize) -> Self {
        self.threshold_tokens = threshold;
        self
    }

    /// Set the number of recent messages to retain.
    #[must_use]
    pub const fn with_retain_recent(mut self, count: usize) -> Self {
        self.retain_recent = count;
        self
    }

    /// Set the minimum messages for compaction.
    #[must_use]
    pub const fn with_min_messages(mut self, count: usize) -> Self {
        self.min_messages_for_compaction = count;
        self
    }

    /// Set whether to auto-compact.
    #[must_use]
    pub const fn with_auto_compact(mut self, auto: bool) -> Self {
        self.auto_compact = auto;
        self
    }

    /// Set the soft cap on tokens kept in the retained tail.
    ///
    /// Bounds [`retain_recent`](Self::retain_recent). See the field docs for
    /// the precise semantics.
    #[must_use]
    pub const fn with_max_retained_tail_tokens(mut self, tokens: usize) -> Self {
        self.max_retained_tail_tokens = tokens;
        self
    }

    /// Set the `max_tokens` budget for the summarization call.
    #[must_use]
    pub const fn with_summary_max_tokens(mut self, tokens: usize) -> Self {
        self.summary_max_tokens = tokens;
        self
    }

    /// Select the compaction implementation.
    #[must_use]
    pub const fn with_engine(mut self, engine: CompactionEngine) -> Self {
        self.engine = engine;
        self
    }

    /// See [`model_aware_threshold`](Self::model_aware_threshold).
    #[must_use]
    pub const fn with_model_aware_threshold(mut self, enabled: bool) -> Self {
        self.model_aware_threshold = enabled;
        self
    }

    /// See [`model_context_reserve_tokens`](Self::model_context_reserve_tokens).
    #[must_use]
    pub const fn with_model_context_reserve_tokens(mut self, tokens: usize) -> Self {
        self.model_context_reserve_tokens = Some(tokens);
        self
    }

    /// See [`model_context_window_tokens`](Self::model_context_window_tokens).
    #[must_use]
    pub const fn with_model_context_window_tokens(mut self, tokens: usize) -> Self {
        self.model_context_window_tokens = Some(tokens);
        self
    }

    /// Whether prune-first compaction and measured host triggers are enabled.
    #[must_use]
    pub const fn uses_prune_first_engine(&self) -> bool {
        matches!(self.engine, CompactionEngine::PruneFirst)
    }

    /// Whether deterministic local Snapcompact rendering is enabled.
    #[must_use]
    pub const fn uses_snapcompact_engine(&self) -> bool {
        matches!(self.engine, CompactionEngine::Snapcompact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CompactionConfig::default();
        assert_eq!(config.threshold_tokens, 80_000);
        assert_eq!(config.retain_recent, 10);
        assert_eq!(config.min_messages_for_compaction, 20);
        assert!(config.auto_compact);
        assert_eq!(
            config.max_retained_tail_tokens,
            DEFAULT_MAX_RETAINED_TAIL_TOKENS
        );
        assert_eq!(config.summary_max_tokens, DEFAULT_SUMMARY_MAX_TOKENS);
        assert_eq!(config.engine, CompactionEngine::Legacy);
        assert!(!config.model_aware_threshold);
        assert_eq!(config.model_context_reserve_tokens, None);
        assert_eq!(config.model_context_window_tokens, None);
    }

    #[test]
    fn test_builder_pattern() {
        let config = CompactionConfig::new()
            .with_threshold_tokens(50_000)
            .with_retain_recent(5)
            .with_min_messages(10)
            .with_auto_compact(false)
            .with_max_retained_tail_tokens(40_000)
            .with_summary_max_tokens(8_000)
            .with_engine(CompactionEngine::PruneFirst)
            .with_model_aware_threshold(true)
            .with_model_context_reserve_tokens(32_000)
            .with_model_context_window_tokens(200_000);

        assert_eq!(config.threshold_tokens, 50_000);
        assert_eq!(config.retain_recent, 5);
        assert_eq!(config.min_messages_for_compaction, 10);
        assert!(!config.auto_compact);
        assert_eq!(config.max_retained_tail_tokens, 40_000);
        assert_eq!(config.summary_max_tokens, 8_000);
        assert_eq!(config.engine, CompactionEngine::PruneFirst);
        assert!(config.model_aware_threshold);
        assert_eq!(config.model_context_reserve_tokens, Some(32_000));
        assert_eq!(config.model_context_window_tokens, Some(200_000));
    }

    #[test]
    fn test_deserialize_without_new_fields_uses_defaults() -> anyhow::Result<()> {
        // Configs serialized before the new knobs existed must still
        // deserialize, falling back to the documented defaults.
        let json = r#"{
            "threshold_tokens": 1234,
            "retain_recent": 7,
            "min_messages_for_compaction": 3,
            "auto_compact": true
        }"#;

        let config: CompactionConfig = serde_json::from_str(json)?;

        assert_eq!(config.threshold_tokens, 1234);
        assert_eq!(config.retain_recent, 7);
        assert_eq!(
            config.max_retained_tail_tokens,
            DEFAULT_MAX_RETAINED_TAIL_TOKENS
        );
        assert_eq!(config.summary_max_tokens, DEFAULT_SUMMARY_MAX_TOKENS);
        assert_eq!(config.engine, CompactionEngine::Legacy);
        assert!(!config.model_aware_threshold);
        assert_eq!(config.model_context_reserve_tokens, None);
        assert_eq!(config.model_context_window_tokens, None);

        Ok(())
    }

    #[test]
    fn model_aware_fields_serde_round_trip() -> anyhow::Result<()> {
        let config = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_model_context_reserve_tokens(24_000)
            .with_model_context_window_tokens(1_048_576);
        let json = serde_json::to_string(&config)?;
        let restored: CompactionConfig = serde_json::from_str(&json)?;
        assert!(restored.model_aware_threshold);
        assert_eq!(restored.model_context_reserve_tokens, Some(24_000));
        assert_eq!(restored.model_context_window_tokens, Some(1_048_576));
        Ok(())
    }

    #[test]
    fn snapcompact_engine_serde_round_trip() -> anyhow::Result<()> {
        let json = serde_json::to_string(&CompactionEngine::Snapcompact)?;
        assert_eq!(json, r#""snapcompact""#);
        assert_eq!(
            serde_json::from_str::<CompactionEngine>(&json)?,
            CompactionEngine::Snapcompact
        );
        assert!(
            CompactionConfig::default()
                .with_engine(CompactionEngine::Snapcompact)
                .uses_snapcompact_engine()
        );
        Ok(())
    }
}
