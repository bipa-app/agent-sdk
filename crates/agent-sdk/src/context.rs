//! Context compaction for long-running conversations.
//!
//! This module provides automatic context compaction to allow conversations
//! to continue without hitting context limits. When the message history
//! grows too large, older messages are summarized using the LLM and
//! replaced with a compact summary.
//!
//! # Overview
//!
//! The compaction system works as follows:
//! 1. Monitor message history size (by token count estimation)
//! 2. When approaching the threshold, partition messages into old and recent
//! 3. Summarize old messages using the LLM
//! 4. Replace history with summary + recent messages
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::{builder, context::CompactionConfig};
//!
//! let agent = builder()
//!     .provider(my_provider)
//!     .with_compaction(CompactionConfig::default())
//!     .build();
//! ```
//!
//! # Configuration
//!
//! Use [`CompactionConfig`] to customize compaction behavior:
//! - `threshold_tokens`: When to trigger compaction
//! - `retain_recent`: How many recent messages to keep intact
//! - `min_messages_for_compaction`: Minimum messages before considering compaction
//! - `auto_compact`: Whether to auto-compact or only on explicit request

mod compactor;
mod config;
mod estimator;

pub use compactor::{ContextCompactor, LlmContextCompactor};
pub use config::CompactionConfig;
pub use estimator::TokenEstimator;
