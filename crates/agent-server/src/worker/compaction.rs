//! Daemon-side auto-compaction integration.
//!
//! The in-process `agent_sdk::agent_loop` already wires
//! [`agent_sdk::context::LlmContextCompactor`] in two places:
//!
//! 1. A **proactive** check in `maybe_compact_messages` that fires
//!    before each LLM call once the staged history exceeds the
//!    configured token threshold.
//! 2. A **reactive** check in `try_recover_prompt_too_long` /
//!    `compact_after_context_overflow` that fires when the provider
//!    rejects a turn with `prompt is too long` (or one of the
//!    sibling shapes the SDK already matches).
//!
//! Until this module landed, the daemon worker (`agent-server`) had
//! neither integration. A long-running thread that crossed the
//! provider's context window — typical when the assistant accumulates
//! tool results turn after turn — surfaced
//! `LLM stream error (kind=InvalidRequest): "prompt is too long: …"`
//! to the user with no recovery path. The host's
//! [`RootTurnDeps::compaction_config`](crate::worker::RootTurnDeps::compaction_config)
//! is now consulted by `execute_root_turn` (pre-call) and
//! `call_llm_with_retry` (post-failure, both private) via the
//! helpers in this module so both topologies share the same compaction
//! contract.
//!
//! # Durability contract
//!
//! Both helpers mutate the durable
//! [`MessageProjectionStore`](crate::journal::message_store::MessageProjectionStore)
//! before they touch the in-memory staged buffer:
//!
//! 1. `MessageProjectionStore::replace_history` rewrites the projection
//!    head transactionally (per the `replace_history` contract in
//!    `crates/agent-server/src/journal/message_store.rs`). After this
//!    write completes, every future attempt — including a fresh attempt
//!    after lease expiry — recovers from the compacted history.
//! 2. The in-memory [`StagedMessageStore`] is then re-pointed to the
//!    same compacted vector via its `replace_history` impl, which now
//!    also resets `seed_len` so the post-compaction commit path
//!    appends only the delta produced *after* the compaction (the
//!    LLM response, the `buffer_turn_messages` fresh user prompt,
//!    etc.) rather than blindly re-appending the whole compacted
//!    history.
//! 3. A [`agent_sdk_core::events::AgentEvent::ContextCompacted`] event
//!    is committed to the durable event repository so subscribers
//!    (TUI, desktop, observability) can surface the compaction in
//!    their transcripts. The event uses the same shape the legacy
//!    in-process loop emits.
//!
//! Step 1 is the load-bearing one: if the host crashes between the
//! projection rewrite and the staged-buffer rewrite, recovery picks
//! up the compacted projection and the next attempt resumes from
//! there. The staged buffer is process-local and discarded across
//! restarts, so a half-applied rewrite is never durable.

use std::sync::Arc;

use agent_sdk::context::{ContextCompactor, LlmContextCompactor};
use agent_sdk_core::events::AgentEvent;
use agent_sdk_providers::LlmProvider;
use agent_sdk_tools::stores::MessageStore;
use anyhow::{Context, Result};
use time::OffsetDateTime;

use crate::journal::staged::StagedMessageStore;

use super::root_turn::RootTurnDeps;

/// Run a pre-call compaction pass against the staged history when the
/// host has wired a [`agent_sdk::context::CompactionConfig`] and the
/// configured threshold is crossed.
///
/// Operates on the **staged history alone** — i.e. the messages that
/// will become the seed of any subsequent commit. It deliberately
/// does *not* peek at the fresh user prompt that `build_chat_request`
/// appends after this helper returns, because that prompt is not yet
/// in the durable projection: folding it into a compaction summary
/// here would cause the commit-time `buffer_turn_messages` append to
/// double-write it. The threshold is therefore evaluated against
/// just-the-staged-history; for the resume path that already includes
/// the prior turn's tool results buffered by `buffer_resume_messages`.
///
/// No-op when `deps.compaction_config` is `None` or
/// `deps.compaction_provider` is `None` (the latter is required to
/// build the [`LlmContextCompactor`] — see
/// [`crate::RootTurnDeps::compaction_provider`]). No-op when the
/// threshold is not crossed.
///
/// # Errors
///
/// Returns an error if the compactor's LLM call fails, or if either
/// the durable projection rewrite or the in-memory staged-buffer
/// rewrite fails. A failed compaction does **not** poison the turn:
/// the caller (`execute_root_turn` / `resume_root_turn`) propagates
/// it as a turn failure so the next attempt can re-try (or, if the
/// problem was transient, recover from the just-compacted projection
/// once a future attempt re-bootstraps).
pub async fn maybe_compact_staged_history(
    deps: &RootTurnDeps<'_>,
    staged_messages: &StagedMessageStore,
    thread_id: &agent_sdk_core::ThreadId,
    now: OffsetDateTime,
) -> Result<()> {
    let Some(cfg) = deps.compaction_config else {
        return Ok(());
    };
    let Some(provider_arc) = deps.compaction_provider else {
        log::debug!(
            "maybe_compact_staged_history: compaction_config set but compaction_provider \
             missing on RootTurnDeps; skipping pre-call check"
        );
        return Ok(());
    };

    let history = staged_messages
        .get_history(thread_id)
        .await
        .context("read staged history for compaction-threshold check")?;

    let compactor =
        LlmContextCompactor::<dyn LlmProvider>::new(Arc::clone(provider_arc), cfg.clone());
    if !compactor.needs_compaction(&history) {
        return Ok(());
    }

    log::info!(
        "Pre-call auto-compaction triggered (thread={thread_id}, message_count={}, \
         threshold_tokens={})",
        history.len(),
        cfg.threshold_tokens,
    );
    apply_compaction(deps, staged_messages, &compactor, history, thread_id, now)
        .await
        .context("pre-call compaction")
}

/// Run a post-failure compaction pass after the provider rejected a
/// turn with `prompt is too long` (or a sibling shape — see
/// [`is_prompt_too_long_error`]).
///
/// Caller is responsible for matching the error first; this helper
/// always attempts compaction when invoked. Returns `Ok(())` after the
/// projection + staged buffer have been rewritten (the caller then
/// rebuilds the chat request from the now-compacted staged history
/// and retries the LLM call), or an error if the compaction itself
/// fails.
///
/// No-op when `deps.compaction_config` is `None` or
/// `deps.compaction_provider` is `None`.
///
/// # Errors
///
/// Returns an error if compaction's LLM call fails or either store
/// rewrite fails. The caller must treat that as a fatal turn error —
/// retrying without rewriting the history would just hit the same
/// `prompt is too long` rejection.
pub async fn compact_after_overflow(
    deps: &RootTurnDeps<'_>,
    staged_messages: &StagedMessageStore,
    thread_id: &agent_sdk_core::ThreadId,
    now: OffsetDateTime,
) -> Result<bool> {
    let Some(cfg) = deps.compaction_config else {
        return Ok(false);
    };
    let Some(provider_arc) = deps.compaction_provider else {
        log::warn!(
            "compact_after_overflow: compaction_config set but compaction_provider missing; \
             cannot recover (thread={thread_id})"
        );
        return Ok(false);
    };

    let history = staged_messages
        .get_history(thread_id)
        .await
        .context("read staged history for overflow recovery")?;
    if history.is_empty() {
        return Ok(false);
    }

    let compactor =
        LlmContextCompactor::<dyn LlmProvider>::new(Arc::clone(provider_arc), cfg.clone());

    log::warn!(
        "Provider rejected turn with prompt-too-long; attempting emergency \
         compaction (thread={thread_id}, message_count={})",
        history.len(),
    );
    apply_compaction(deps, staged_messages, &compactor, history, thread_id, now)
        .await
        .context("overflow recovery compaction")?;
    Ok(true)
}

/// Inner shared body — runs the compactor, rewrites the projection,
/// rewrites the staged buffer, emits the `ContextCompacted` event.
async fn apply_compaction(
    deps: &RootTurnDeps<'_>,
    staged_messages: &StagedMessageStore,
    compactor: &LlmContextCompactor<dyn LlmProvider>,
    history: Vec<agent_sdk_core::llm::Message>,
    thread_id: &agent_sdk_core::ThreadId,
    now: OffsetDateTime,
) -> Result<()> {
    let result = compactor
        .compact_history(history)
        .await
        .context("compactor.compact_history failed")?;

    deps.message_store
        .replace_history(thread_id, result.messages.clone(), now)
        .await
        .context("replace durable projection history")?;

    staged_messages
        .replace_history(thread_id, result.messages.clone())
        .await
        .context("replace staged buffer history")?;

    let event = AgentEvent::context_compacted(
        result.original_count,
        result.new_count,
        result.original_tokens,
        result.new_tokens,
    );
    let committed = deps
        .event_repo
        .commit_event(thread_id, event, now)
        .await
        .context("commit ContextCompacted event")?;
    deps.event_notifier.notify(std::slice::from_ref(&committed));

    log::info!(
        "Auto-compaction complete (thread={thread_id}, \
         original_count={}, new_count={}, original_tokens={}, new_tokens={})",
        result.original_count,
        result.new_count,
        result.original_tokens,
        result.new_tokens,
    );
    Ok(())
}

/// True when an error message indicates the prompt exceeds the model's context window.
///
/// Mirrors the legacy SDK's matcher in
/// `agent-sdk/src/agent_loop/turn.rs::is_prompt_too_long_error` so
/// the daemon and in-process loops recover from the same provider
/// vocabulary.
#[must_use]
pub fn is_prompt_too_long_error(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    lower.contains("prompt is too long")
        || lower.contains("maximum context length")
        || lower.contains("context_length_exceeded")
        || lower.contains("input is too long")
        || lower.contains("request too large")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_too_long_matcher_covers_known_shapes() {
        // Anthropic 1M
        assert!(is_prompt_too_long_error(
            "prompt is too long: 1010596 tokens > 1000000 maximum"
        ));
        // OpenAI
        assert!(is_prompt_too_long_error(
            "context_length_exceeded: This model's maximum context length is 8192 tokens"
        ));
        assert!(is_prompt_too_long_error(
            "Maximum context length 4096 tokens exceeded"
        ));
        // Gemini
        assert!(is_prompt_too_long_error("Input is too long for this model"));
        // Bedrock
        assert!(is_prompt_too_long_error("Request too large for the model"));

        // Negatives
        assert!(!is_prompt_too_long_error("rate limited"));
        assert!(!is_prompt_too_long_error("transport error"));
        assert!(!is_prompt_too_long_error(""));
    }
}
