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
//! Both helpers append a durable compaction entry before they touch the
//! in-memory staged buffer:
//!
//! 1. `MessageProjectionStore::append_compaction` preserves every committed
//!    message, folds any recovery draft into that raw transcript, and appends a
//!    range-addressed replacement prefix atomically.
//! 2. `MessageProjectionStore::get_history` rebuilds the effective LLM view
//!    from the latest replacement prefix plus the retained raw tail. The
//!    in-memory [`StagedMessageStore`] is re-pointed to that same view and resets
//!    `seed_len`, so the post-compaction commit appends only later deltas.
//! 3. A [`agent_sdk_foundation::events::AgentEvent::ContextCompacted`] event is
//!    committed so renderers can collapse the compacted span while the full raw
//!    transcript remains addressable through the projection snapshot.
//!
//! If the host crashes after step 1, recovery rebuilds the compacted LLM view
//! from the append-only projection. The staged buffer is process-local and is
//! discarded across restarts.

use std::sync::Arc;

use agent_sdk::context::{
    CompactionPurpose, ContextCompactor, FailedCompaction, LlmContextCompactor,
};
use agent_sdk_foundation::TokenUsage;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_providers::LlmProvider;
use agent_sdk_tools::stores::MessageStore;
use anyhow::{Context, Result};
use time::OffsetDateTime;

use crate::journal::staged::StagedMessageStore;
use crate::journal::turn_attempt::TurnAttemptOutcome;

use super::root_turn::RootTurnDeps;

#[derive(Debug, Clone)]
pub(crate) struct CompactionOutcome {
    pub(crate) completed: bool,
    pub(crate) applied: bool,
    pub(crate) llm_usage: TokenUsage,
}

impl CompactionOutcome {
    fn not_run() -> Self {
        Self {
            completed: false,
            applied: false,
            llm_usage: TokenUsage::default(),
        }
    }
}
#[derive(Debug)]
pub(crate) struct CompactionFailure {
    pub(crate) error: anyhow::Error,
    pub(crate) llm_usage: TokenUsage,
}

impl CompactionFailure {
    fn with_context(self, context: &'static str) -> Self {
        Self {
            error: self.error.context(context),
            llm_usage: self.llm_usage,
        }
    }
}

impl From<anyhow::Error> for CompactionFailure {
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            llm_usage: TokenUsage::default(),
        }
    }
}

impl std::fmt::Display for CompactionFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, formatter)
    }
}

impl std::error::Error for CompactionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

impl From<FailedCompaction> for CompactionFailure {
    fn from(failure: FailedCompaction) -> Self {
        Self {
            error: failure.error,
            llm_usage: failure.llm_usage,
        }
    }
}

type CompactionResult<T> = std::result::Result<T, CompactionFailure>;
struct MeasuredAnchor {
    tokens: usize,
    request_message_count: Option<usize>,
}

fn trigger_tokens(measured: Option<usize>, estimated_fallback: usize) -> (usize, &'static str) {
    measured.map_or((estimated_fallback, "estimated_fallback"), |tokens| {
        (tokens, "measured")
    })
}

/// Latest successful billed attempt, ignoring anything billed before the
/// newest durable compaction boundary: those attempts measured a history
/// that no longer exists, and trusting them would re-trigger compaction on
/// an already-compacted (or prune-only-compacted) projection.
async fn latest_measured_anchor(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    ignore_closed_at_or_before: Option<OffsetDateTime>,
) -> Result<Option<MeasuredAnchor>> {
    let tasks = deps
        .task_store
        .list_by_thread(thread_id)
        .await
        .context("list thread tasks for measured compaction trigger")?;
    let mut latest: Option<(OffsetDateTime, MeasuredAnchor)> = None;
    for task in tasks {
        let attempts = deps
            .attempt_store
            .list_by_task(&task.id)
            .await
            .context("list turn attempts for measured compaction trigger")?;
        for attempt in attempts {
            if attempt.outcome != Some(TurnAttemptOutcome::Success) {
                continue;
            }
            let Some(closed_at) = attempt.closed_at else {
                continue;
            };
            if ignore_closed_at_or_before.is_some_and(|fence| closed_at <= fence) {
                continue;
            }
            let total = u64::from(attempt.input_tokens.unwrap_or(0))
                .saturating_add(u64::from(attempt.output_tokens.unwrap_or(0)))
                .saturating_add(u64::from(attempt.cached_input_tokens.unwrap_or(0)))
                .saturating_add(u64::from(attempt.cache_creation_input_tokens.unwrap_or(0)));
            if total == 0 {
                continue;
            }
            let tokens = usize::try_from(total).map_or(usize::MAX, |value| value);
            if latest
                .as_ref()
                .is_none_or(|(latest_at, _)| closed_at > *latest_at)
            {
                let request_message_count = attempt
                    .request_blob
                    .get("messages")
                    .and_then(serde_json::Value::as_array)
                    .map(Vec::len);
                latest = Some((
                    closed_at,
                    MeasuredAnchor {
                        tokens,
                        request_message_count,
                    },
                ));
            }
        }
    }
    Ok(latest.map(|(_, anchor)| anchor))
}

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
/// Cancellation while the summarisation call is in flight is a successful
/// no-op: no durable projection, staged history, or event is mutated.
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
pub(crate) async fn maybe_compact_staged_history(
    deps: &RootTurnDeps<'_>,
    staged_messages: &StagedMessageStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
) -> CompactionResult<CompactionOutcome> {
    // Cooperative cancellation: skip the (billed) summarisation LLM call
    // when the root turn has already been cancelled.
    if deps.is_cancelled() {
        return Ok(CompactionOutcome::not_run());
    }
    let Some(cfg) = deps.compaction_config else {
        return Ok(CompactionOutcome::not_run());
    };
    let Some(provider_arc) = deps.compaction_provider else {
        log::debug!(
            "maybe_compact_staged_history: compaction_config set but compaction_provider \
             missing on RootTurnDeps; skipping pre-call check"
        );
        return Ok(CompactionOutcome::not_run());
    };

    let history = staged_messages
        .get_history(thread_id)
        .await
        .context("read staged history for compaction-threshold check")?;

    let compactor =
        LlmContextCompactor::<dyn LlmProvider>::new(Arc::clone(provider_arc), cfg.clone());
    let compactor = if cfg.uses_prune_first_engine() {
        compactor.with_purpose(CompactionPurpose::PreSpawn)
    } else {
        compactor
    };
    let compactor = if let Some(store) = deps.compaction_artifact_store {
        compactor.with_artifact_store(Arc::clone(store))
    } else {
        compactor
    };
    let compactor = if let Some(cancel) = deps.cancel {
        compactor.with_cancellation(cancel.clone())
    } else {
        compactor
    };
    if !cfg.auto_compact || history.len() < cfg.min_messages_for_compaction {
        return Ok(CompactionOutcome::not_run());
    }
    let (trigger_tokens, trigger_source) =
        if cfg.uses_prune_first_engine() || cfg.uses_snapcompact_engine() {
            let last_compaction_at = deps
                .message_store
                .get(thread_id)
                .await
                .context("read projection for measured-trigger compaction fence")?
                .and_then(|projection| {
                    projection
                        .compactions
                        .last()
                        .map(|boundary| boundary.created_at)
                });
            let measured = latest_measured_anchor(deps, thread_id, last_compaction_at)
                .await?
                .map(|anchor| {
                    let appended = anchor
                        .request_message_count
                        .and_then(|count| history.get(count..))
                        .map_or(0, |suffix| compactor.estimate_tokens(suffix));
                    anchor.tokens.saturating_add(appended)
                });
            trigger_tokens(measured, compactor.estimate_tokens(&history))
        } else {
            (compactor.estimate_tokens(&history), "legacy_estimated")
        };
    if trigger_tokens <= cfg.threshold_tokens {
        return Ok(CompactionOutcome::not_run());
    }

    log::info!(
        "Pre-call auto-compaction triggered (thread={thread_id}, message_count={}, \
         trigger_tokens={trigger_tokens}, trigger_source={trigger_source}, threshold_tokens={}, \
         engine={:?}, purpose=pre_spawn)",
        history.len(),
        cfg.threshold_tokens,
        cfg.engine,
    );
    apply_compaction(deps, staged_messages, &compactor, history, thread_id, now)
        .await
        .map_err(|failure| failure.with_context("pre-call compaction"))
}

/// Run a post-failure compaction pass after the provider rejected a
/// turn with `prompt is too long` (or a sibling shape — see
/// [`is_prompt_too_long_error`]).
///
/// Caller is responsible for matching the error first; this helper
/// always attempts compaction when invoked. The outcome carries whether the
/// projection was applied and the exact provider-billed usage. An unapplied
/// result means configuration was absent or cancellation won before the
/// durable append.
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
pub(crate) async fn compact_after_overflow(
    deps: &RootTurnDeps<'_>,
    staged_messages: &StagedMessageStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
) -> CompactionResult<CompactionOutcome> {
    // Cooperative cancellation: skip emergency compaction (and its LLM
    // call) on an already-cancelled turn; the caller bails on `false`.
    if deps.is_cancelled() {
        return Ok(CompactionOutcome::not_run());
    }
    let Some(cfg) = deps.compaction_config else {
        return Ok(CompactionOutcome::not_run());
    };
    let Some(provider_arc) = deps.compaction_provider else {
        log::warn!(
            "compact_after_overflow: compaction_config set but compaction_provider missing; \
             cannot recover (thread={thread_id})"
        );
        return Ok(CompactionOutcome::not_run());
    };

    let history = staged_messages
        .get_history(thread_id)
        .await
        .context("read staged history for overflow recovery")?;
    if history.is_empty() {
        return Ok(CompactionOutcome::not_run());
    }

    let compactor =
        LlmContextCompactor::<dyn LlmProvider>::new(Arc::clone(provider_arc), cfg.clone());
    let compactor = if cfg.uses_prune_first_engine() {
        compactor.with_purpose(CompactionPurpose::Overflow)
    } else {
        compactor
    };
    let compactor = if let Some(store) = deps.compaction_artifact_store {
        compactor.with_artifact_store(Arc::clone(store))
    } else {
        compactor
    };
    let compactor = if let Some(cancel) = deps.cancel {
        compactor.with_cancellation(cancel.clone())
    } else {
        compactor
    };

    log::warn!(
        "Provider rejected turn with prompt-too-long; attempting emergency \
         compaction (thread={thread_id}, message_count={}, engine={:?}, purpose=overflow)",
        history.len(),
        cfg.engine,
    );
    apply_compaction(deps, staged_messages, &compactor, history, thread_id, now)
        .await
        .map_err(|failure| failure.with_context("overflow recovery compaction"))
}

/// Inner shared body — runs the compactor and returns `false` if cancellation
/// wins. An applied result appends the projection compaction, rewrites the
/// staged buffer, and emits the `ContextCompacted` event.
///
/// The cancel arm drops the compaction future, but the Snapcompact
/// `spawn_blocking` worker it started keeps running detached. The callers
/// wire the same [`RootTurnDeps::cancel`] token into the compactor via
/// `with_cancellation`, so the already-tripped token doubles as a publish
/// fence inside that worker: no artifact batch is published after this arm
/// wins, instead of the batch landing as an unreferenced orphan.
async fn apply_compaction(
    deps: &RootTurnDeps<'_>,
    staged_messages: &StagedMessageStore,
    compactor: &LlmContextCompactor<dyn LlmProvider>,
    history: Vec<agent_sdk_foundation::llm::Message>,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
) -> CompactionResult<CompactionOutcome> {
    let compact = compactor.compact_history_with_usage(history);
    let completed = if let Some(cancel) = deps.cancel {
        tokio::select! {
            biased;
            () = cancel.cancelled() => return Ok(CompactionOutcome::not_run()),
            result = compact => result,
        }
    } else {
        compact.await
    };

    let result = match completed {
        Ok(result) => {
            deps.note_token_usage(&result.llm_usage);
            result
        }
        Err(failure) => {
            deps.note_token_usage(&failure.llm_usage);
            return Err(CompactionFailure::from(failure)
                .with_context("compactor.compact_history_with_usage failed"));
        }
    };
    let llm_usage = result.llm_usage.clone();

    // Provider usage becomes live-accounted before cancellation or any
    // durability decision. If cancellation completed in the same scheduler
    // turn as the response, the attempt still owns these billed tokens while
    // projection, staged history, and events remain untouched.
    if deps.is_cancelled() {
        return Ok(CompactionOutcome {
            completed: true,
            applied: false,
            llm_usage,
        });
    }

    deps.message_store
        .append_compaction(
            thread_id,
            result.messages.clone(),
            result.original_count,
            result.retained_count,
            now,
        )
        .await
        .context("append durable compaction entry")
        .map_err(|error| CompactionFailure {
            error,
            llm_usage: llm_usage.clone(),
        })?;

    staged_messages
        .replace_history(thread_id, result.messages.clone())
        .await
        .context("replace staged buffer history")
        .map_err(|error| CompactionFailure {
            error,
            llm_usage: llm_usage.clone(),
        })?;

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
        .context("commit ContextCompacted event")
        .map_err(|error| CompactionFailure {
            error,
            llm_usage: llm_usage.clone(),
        })?;
    deps.event_notifier.notify(std::slice::from_ref(&committed));

    log::info!(
        "Auto-compaction complete (thread={thread_id}, \
         original_count={}, new_count={}, original_tokens={}, new_tokens={})",
        result.original_count,
        result.new_count,
        result.original_tokens,
        result.new_tokens,
    );
    Ok(CompactionOutcome {
        completed: true,
        applied: true,
        llm_usage,
    })
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
        || lower.contains("exceeds the context window")
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
        // OpenAI Responses API (`response.failed` prose)
        assert!(is_prompt_too_long_error(
            "Your input exceeds the context window of this model. \
             Please adjust your input and try again."
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

    #[test]
    fn measured_usage_prevents_large_window_estimator_regression() {
        let (tokens, source) = trigger_tokens(Some(88_000), 1_200_000);
        assert_eq!(tokens, 88_000);
        assert_eq!(source, "measured");
        assert!(tokens <= 1_000_000);
    }

    #[test]
    fn estimator_is_used_only_before_measured_usage_exists() {
        let (tokens, source) = trigger_tokens(None, 120_000);
        assert_eq!(tokens, 120_000);
        assert_eq!(source, "estimated_fallback");
    }
}
