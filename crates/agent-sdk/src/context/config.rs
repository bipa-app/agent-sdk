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

const fn default_max_retained_tail_tokens() -> usize {
    DEFAULT_MAX_RETAINED_TAIL_TOKENS
}

const fn default_summary_max_tokens() -> usize {
    DEFAULT_SUMMARY_MAX_TOKENS
}

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
    }

    #[test]
    fn test_builder_pattern() {
        let config = CompactionConfig::new()
            .with_threshold_tokens(50_000)
            .with_retain_recent(5)
            .with_min_messages(10)
            .with_auto_compact(false)
            .with_max_retained_tail_tokens(40_000)
            .with_summary_max_tokens(8_000);

        assert_eq!(config.threshold_tokens, 50_000);
        assert_eq!(config.retain_recent, 5);
        assert_eq!(config.min_messages_for_compaction, 10);
        assert!(!config.auto_compact);
        assert_eq!(config.max_retained_tail_tokens, 40_000);
        assert_eq!(config.summary_max_tokens, 8_000);
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

        Ok(())
    }
}
