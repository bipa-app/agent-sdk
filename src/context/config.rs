//! Configuration for context compaction.

use serde::{Deserialize, Serialize};

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
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            threshold_tokens: 80_000,
            retain_recent: 10,
            min_messages_for_compaction: 20,
            auto_compact: true,
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
    }

    #[test]
    fn test_builder_pattern() {
        let config = CompactionConfig::new()
            .with_threshold_tokens(50_000)
            .with_retain_recent(5)
            .with_min_messages(10)
            .with_auto_compact(false);

        assert_eq!(config.threshold_tokens, 50_000);
        assert_eq!(config.retain_recent, 5);
        assert_eq!(config.min_messages_for_compaction, 10);
        assert!(!config.auto_compact);
    }
}
