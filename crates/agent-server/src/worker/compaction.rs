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
    CompactionConfig, CompactionPurpose, ContextCompactor, DEFAULT_MODEL_CONTEXT_RESERVE_TOKENS,
    FailedCompaction, LlmContextCompactor,
};
use agent_sdk_foundation::TokenUsage;
use agent_sdk_foundation::events::{AgentEvent, CompactionTrigger};
use agent_sdk_foundation::llm::{ChatRequest, ThinkingConfig, ThinkingMode};
use agent_sdk_providers::{DefaultReasoning, LlmProvider, ModelCapabilities};
use agent_sdk_tools::stores::MessageStore;
use anyhow::{Context, Result};
use time::OffsetDateTime;

use crate::journal::staged::StagedMessageStore;
use crate::journal::turn_attempt::{TurnAttempt, TurnAttemptOutcome};

use super::root_turn::RootTurnDeps;

#[derive(Debug, Clone)]
pub(crate) struct CompactionOutcome {
    pub(crate) completed: bool,
    pub(crate) applied: bool,
    pub(crate) llm_usage: TokenUsage,
    /// Summarizer response model; `None` when no summarizer call ran.
    pub(crate) response_model: Option<String>,
    pub(crate) served_route: Option<String>,
}

impl CompactionOutcome {
    fn not_run() -> Self {
        Self {
            completed: false,
            applied: false,
            llm_usage: TokenUsage::default(),
            response_model: None,
            served_route: None,
        }
    }
}
#[derive(Debug)]
pub(crate) struct CompactionFailure {
    pub(crate) error: anyhow::Error,
    pub(crate) llm_usage: TokenUsage,
    pub(crate) response_model: Option<String>,
    pub(crate) served_route: Option<String>,
}

impl CompactionFailure {
    fn with_context(self, context: &'static str) -> Self {
        Self {
            error: self.error.context(context),
            ..self
        }
    }
}

impl From<anyhow::Error> for CompactionFailure {
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            llm_usage: TokenUsage::default(),
            response_model: None,
            served_route: None,
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
            response_model: failure.response_model,
            served_route: failure.served_route,
        }
    }
}

type CompactionResult<T> = std::result::Result<T, CompactionFailure>;
struct MeasuredAnchor {
    tokens: usize,
    request_message_count: usize,
}

/// SHA-256 over the provider identity and every non-message
/// [`ChatRequest`] field, hex-encoded. Request-only reminders ride inside
/// the final message, so they are message content too.
///
/// Measured input also prices the system prompt, tools, and every other
/// envelope field, so an anchor is only valid for a request with the same
/// envelope. The destructure is exhaustive on purpose: a new
/// [`ChatRequest`] field must fail compilation here until it is either
/// fingerprinted or explicitly excluded.
pub(crate) fn request_envelope_fingerprint(
    provider: &str,
    model: &str,
    request: &ChatRequest,
) -> Result<String> {
    use sha2::{Digest, Sha256};
    let ChatRequest {
        system,
        messages: _,
        tools,
        max_tokens,
        max_tokens_explicit,
        session_id,
        cached_content,
        thinking,
        tool_choice,
        response_format,
        cache,
        request_only_reminders: _,
    } = request;
    let canonical = serde_json::to_vec(&(
        provider,
        model,
        system,
        tools,
        max_tokens,
        max_tokens_explicit,
        session_id,
        cached_content,
        thinking,
        tool_choice,
        response_format,
        cache,
    ))
    .context("serialize request envelope for fingerprint")?;
    let digest = Sha256::digest(&canonical);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

struct AnchorRequirements<'a> {
    /// Durable `history_generation`; `compactions.len()` cannot serve as
    /// the lineage because `replace_history` clears the compaction list.
    current_history_generation: u64,
    history_len: usize,
    expected_request_fingerprint: &'a str,
    /// `Some` for a provider without a stable request route: only a request
    /// whose stream confirmed this route and model served it measured the
    /// backend the next call dispatches to first.
    required_served_identity: Option<ServedIdentity<'a>>,
}

/// The `(route, model)` pair a stream's `Done` stamps. A route alone can
/// serve several models, so an anchor must match both.
#[derive(Clone, Copy)]
struct ServedIdentity<'a> {
    route: &'a str,
    model: &'a str,
}

/// `request_blob.user_prompt` of internal context-compaction attempts.
pub(crate) const COMPACTION_AUDIT_PROMPT: &str = "<context-compaction>";

fn is_compaction_audit_attempt(attempt: &TurnAttempt) -> bool {
    attempt
        .response_blob
        .as_ref()
        .and_then(|blob| blob.get("operation"))
        .and_then(serde_json::Value::as_str)
        == Some("context_compaction")
}

/// Measured occupancy of one settled main-call attempt, or `None` when the
/// row cannot anchor the trigger.
///
/// Occupancy is the stamped single-request `request_input_tokens`, never
/// the aggregate `input_tokens` column: that column folds abandoned
/// failover/refresh attempts, and the cache counters are subsets of it.
/// Compaction audit rows bill the summarizer's own request and never
/// anchor. The stamped generation, envelope fingerprint, and message count
/// must all match the request about to be sent.
fn attempt_measured_anchor(
    attempt: &TurnAttempt,
    requirements: &AnchorRequirements<'_>,
) -> Option<(OffsetDateTime, MeasuredAnchor)> {
    if attempt.outcome != Some(TurnAttemptOutcome::Success) {
        return None;
    }
    attempt.closed_at?;
    if is_compaction_audit_attempt(attempt) {
        return None;
    }
    let blob = attempt.response_blob.as_ref()?;
    let stamped_generation = blob
        .get("history_generation")
        .and_then(serde_json::Value::as_u64)?;
    if stamped_generation != requirements.current_history_generation {
        return None;
    }
    if blob
        .get("request_fingerprint")
        .and_then(serde_json::Value::as_str)
        != Some(requirements.expected_request_fingerprint)
    {
        return None;
    }
    if let Some(required) = requirements.required_served_identity
        && (blob.get("served_route").and_then(serde_json::Value::as_str) != Some(required.route)
            || blob.get("served_model").and_then(serde_json::Value::as_str) != Some(required.model))
    {
        return None;
    }
    let request_message_count = blob
        .get("request_message_count")
        .and_then(serde_json::Value::as_u64)
        .and_then(|count| usize::try_from(count).ok())
        .filter(|count| *count <= requirements.history_len)?;
    let input_tokens = blob
        .get("request_input_tokens")
        .and_then(serde_json::Value::as_u64)
        .filter(|tokens| *tokens > 0)?;
    Some((
        attempt.opened_at,
        MeasuredAnchor {
            tokens: usize::try_from(input_tokens).unwrap_or(usize::MAX),
            request_message_count,
        },
    ))
}

/// The full-history estimate is lazy: it only runs without an anchor.
fn trigger_tokens(
    measured: Option<usize>,
    estimated_fallback: impl FnOnce() -> usize,
) -> (usize, &'static str) {
    measured.map_or_else(
        || (estimated_fallback(), "estimated_fallback"),
        |tokens| (tokens, "measured"),
    )
}

/// Latest eligible main-call anchor. Histories only grow between rewrites,
/// so the greatest covered message count names the latest request;
/// `opened_at` only breaks ties.
async fn latest_measured_anchor(
    deps: &RootTurnDeps<'_>,
    thread_id: &agent_sdk_foundation::ThreadId,
    requirements: &AnchorRequirements<'_>,
) -> Result<Option<MeasuredAnchor>> {
    let tasks = deps
        .task_store
        .list_by_thread(thread_id)
        .await
        .context("list thread tasks for measured compaction trigger")?;
    let mut best: Option<(usize, OffsetDateTime, MeasuredAnchor)> = None;
    for task in tasks {
        let attempts = deps
            .attempt_store
            .list_by_task(&task.id)
            .await
            .context("list turn attempts for measured compaction trigger")?;
        for attempt in attempts {
            let Some((opened_at, anchor)) = attempt_measured_anchor(&attempt, requirements) else {
                continue;
            };
            let key = (anchor.request_message_count, opened_at);
            if best
                .as_ref()
                .is_none_or(|(count, at, _)| key > (*count, *at))
            {
                best = Some((anchor.request_message_count, opened_at, anchor));
            }
        }
    }
    Ok(best.map(|(_, _, anchor)| anchor))
}

fn saturating_usize(tokens: u32) -> usize {
    usize::try_from(tokens).unwrap_or(usize::MAX)
}

/// Visible output may claim at most this fraction of the serving window, so
/// at least three quarters stay available to the prompt and history.
const OUTPUT_WINDOW_SHARE_DIVISOR: u32 = 4;

/// Floor of the visible-output window share, so small-window models keep the
/// output budget the Anthropic and Vertex providers default to when a caller
/// sets none.
const MIN_OUTPUT_WINDOW_SHARE_TOKENS: u32 = 32_000;

/// Output a request may spend on reasoning on top of its visible share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReasoningAllowance {
    /// The request configures no reasoning.
    None,
    /// An explicit thinking budget.
    Budget(u32),
    /// Reasoning sized by effort or by the model (configured without a
    /// budget, or on by default), which may take as much as the visible
    /// share.
    Unbudgeted,
}

/// How [`serving_max_tokens`] bounds a host's output budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputSizing {
    /// Only the model's registered `max_output_tokens`.
    ModelMax,
    /// Also the visible-output share of `window`, plus the reasoning
    /// allowance.
    WindowShare {
        window: u32,
        reasoning: ReasoningAllowance,
    },
}

impl OutputSizing {
    /// The window share only pays off where a model-aware threshold drives
    /// the pre-call trigger, and it shares the window that threshold uses.
    /// It also needs the serving model's capability row: without one (an
    /// unknown model, or a route that may dispatch elsewhere and hides the
    /// served provider's thinking) the request's reasoning is unknown and a
    /// smaller request could starve it. `thinking` is the configuration the
    /// provider will send: the request's, else the provider's own.
    /// `provider_reasons` is the provider's own effort-only reasoning
    /// setting, which applies when `thinking` is `None`.
    pub(crate) fn for_request(
        trigger: Option<ModelAwareTrigger<'_>>,
        capabilities: Option<&ModelCapabilities>,
        thinking: Option<&ThinkingConfig>,
        provider_reasons: bool,
    ) -> Self {
        let (Some(trigger), Some(capabilities)) = (trigger, capabilities) else {
            return Self::ModelMax;
        };
        let Some(window) = effective_window_tokens(trigger.0, Some(capabilities)) else {
            return Self::ModelMax;
        };
        let reasoning = match thinking.map(|thinking| &thinking.mode) {
            None if provider_reasons => ReasoningAllowance::Unbudgeted,
            None => match capabilities.default_reasoning {
                DefaultReasoning::On => ReasoningAllowance::Unbudgeted,
                DefaultReasoning::Off => ReasoningAllowance::None,
            },
            Some(ThinkingMode::Enabled { budget_tokens }) => {
                ReasoningAllowance::Budget(*budget_tokens)
            }
            Some(ThinkingMode::Adaptive | ThinkingMode::Default) => ReasoningAllowance::Unbudgeted,
        };
        Self::WindowShare {
            window: u32::try_from(window).unwrap_or(u32::MAX),
            reasoning,
        }
    }
}

/// A compaction config whose model-aware threshold drives the pre-call
/// trigger; only [`model_aware_trigger`] builds one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelAwareTrigger<'a>(&'a CompactionConfig);

/// `cfg` when its model-aware threshold drives the pre-call trigger:
/// automatic compaction is on and a compaction provider can run it. The
/// message-count minimum is not consulted, so a request's size, and with it
/// the envelope fingerprint, never changes as history grows.
pub(crate) fn model_aware_trigger(
    cfg: Option<&CompactionConfig>,
    has_compaction_provider: bool,
) -> Option<ModelAwareTrigger<'_>> {
    cfg.filter(|cfg| has_compaction_provider && cfg.auto_compact && cfg.model_aware_threshold)
        .map(ModelAwareTrigger)
}

/// The window the model-aware threshold derives from: the configured
/// override, else the serving model's registered window.
fn effective_window_tokens(
    cfg: &CompactionConfig,
    capabilities: Option<&ModelCapabilities>,
) -> Option<usize> {
    cfg.model_context_window_tokens
        .or_else(|| {
            capabilities
                .and_then(|caps| caps.context_window)
                .map(saturating_usize)
        })
        .filter(|window| *window > 0)
}

/// Capabilities of the model that will serve the next request. A dynamic
/// dispatcher (failover chain, per-request router) may serve the call from a
/// route its representative capabilities do not describe, so it contributes
/// none.
pub(crate) fn serving_capabilities(
    provider: &dyn LlmProvider,
) -> Option<&'static ModelCapabilities> {
    if provider.has_stable_request_route() {
        provider.capabilities()
    } else {
        None
    }
}

/// `max_tokens` to request from the serving model for a host-requested
/// output budget.
///
/// The request is always capped at the model's registered
/// `max_output_tokens`, above which the provider rejects the call. Under
/// [`OutputSizing::WindowShare`] it is also capped at a quarter of the
/// window, floored at [`MIN_OUTPUT_WINDOW_SHARE_TOKENS`], plus the
/// reasoning allowance, because every output token reserved is an input
/// token the compaction trigger must give up; reasoning rides on top so it
/// never eats into the visible share. A request with neither bound keeps
/// its budget.
pub(crate) fn serving_max_tokens(
    requested: u32,
    capabilities: Option<&ModelCapabilities>,
    sizing: OutputSizing,
) -> u32 {
    let model_max = capabilities
        .and_then(|caps| caps.max_output_tokens)
        .filter(|tokens| *tokens > 0);
    let window_share = match sizing {
        OutputSizing::ModelMax => None,
        OutputSizing::WindowShare { window, reasoning } => {
            let share = (window / OUTPUT_WINDOW_SHARE_DIVISOR).max(MIN_OUTPUT_WINDOW_SHARE_TOKENS);
            let allowance = match reasoning {
                ReasoningAllowance::None => 0,
                ReasoningAllowance::Budget(budget) => budget,
                ReasoningAllowance::Unbudgeted => share,
            };
            Some(share.saturating_add(allowance))
        }
    };
    [model_max, window_share]
        .into_iter()
        .flatten()
        .fold(requested, u32::min)
}

/// Trigger threshold for one call; see
/// [`CompactionConfig::model_aware_threshold`]. The row's
/// `max_output_tokens` reserves only what [`serving_max_tokens`] would
/// request of it, and a zero `requested_max_tokens` floors the reserve at
/// that amount instead. A reserve that consumes the whole window leaves
/// `threshold_tokens` unchanged.
fn effective_threshold_tokens(
    cfg: &CompactionConfig,
    capabilities: Option<&ModelCapabilities>,
    requested_max_tokens: u32,
) -> (usize, &'static str) {
    if !cfg.model_aware_threshold {
        return (cfg.threshold_tokens, "fixed");
    }
    let Some(window) = effective_window_tokens(cfg, capabilities) else {
        return (cfg.threshold_tokens, "fixed_unknown_model");
    };
    let row_max_output = capabilities
        .and_then(|caps| caps.max_output_tokens)
        .map(|tokens| {
            saturating_usize(serving_max_tokens(
                tokens,
                capabilities,
                OutputSizing::WindowShare {
                    window: u32::try_from(window).unwrap_or(u32::MAX),
                    reasoning: ReasoningAllowance::None,
                },
            ))
        });
    let requested_output = Some(requested_max_tokens)
        .filter(|tokens| *tokens > 0)
        .map(saturating_usize)
        .or(row_max_output)
        .unwrap_or(0);
    let reserve = cfg
        .model_context_reserve_tokens
        .or(row_max_output)
        .unwrap_or(DEFAULT_MODEL_CONTEXT_RESERVE_TOKENS)
        .max(requested_output);
    window
        .checked_sub(reserve)
        .filter(|threshold| *threshold > 0)
        .map_or(
            (cfg.threshold_tokens, "fixed_invalid_model_window"),
            |threshold| (threshold, "model_aware"),
        )
}

/// Threshold for a call served by `main_provider`; see
/// [`serving_capabilities`].
fn serving_threshold_tokens(
    cfg: &CompactionConfig,
    main_provider: &dyn LlmProvider,
    requested_max_tokens: u32,
) -> (usize, &'static str) {
    effective_threshold_tokens(
        cfg,
        serving_capabilities(main_provider),
        requested_max_tokens,
    )
}

/// The compactor's own budgets (prune-only acceptance, Snapcompact frame
/// limits) read `threshold_tokens`, so it must carry the effective value.
fn build_compactor(
    deps: &RootTurnDeps<'_>,
    provider: &Arc<dyn LlmProvider>,
    cfg: &CompactionConfig,
    threshold_tokens: usize,
    purpose: CompactionPurpose,
) -> LlmContextCompactor<dyn LlmProvider> {
    let compactor = LlmContextCompactor::<dyn LlmProvider>::new(
        Arc::clone(provider),
        cfg.clone().with_threshold_tokens(threshold_tokens),
    )
    .with_provisional_artifacts();
    let compactor = if cfg.uses_prune_first_engine() {
        compactor.with_purpose(purpose)
    } else {
        compactor
    };
    let compactor = if let Some(store) = deps.compaction_artifact_store {
        compactor.with_artifact_store(Arc::clone(store))
    } else {
        compactor
    };
    if let Some(cancel) = deps.cancel {
        compactor.with_cancellation(cancel.clone())
    } else {
        compactor
    }
}

/// A pre-call compaction whose trigger crossed the threshold, planned before
/// any audit row exists so the caller can bracket the summarizer call with a
/// dedicated `<context-compaction>` attempt.
pub(crate) struct PlannedCompaction {
    compactor: LlmContextCompactor<dyn LlmProvider>,
    history: Vec<agent_sdk_foundation::llm::Message>,
}

/// Decide, without side effects, whether a pre-call compaction should run.
///
/// Only the staged history is ever summarized; the fresh user prompt is not
/// in the durable projection yet, so folding it in would make the commit
/// double-write it. Its estimate (`pending_input_tokens`) still counts
/// toward the trigger because it rides on the next request.
///
/// Trigger occupancy is the latest eligible measured anchor plus the
/// estimated staged suffix it did not cover; the full history is estimated
/// only when no anchor exists. A provider without a stable request route
/// anchors only on requests its own representative route and model served.
/// `expected_request` is the message-less envelope of the request about to
/// be sent.
pub(crate) async fn plan_pre_call_compaction(
    deps: &RootTurnDeps<'_>,
    main_provider: &dyn LlmProvider,
    staged_messages: &StagedMessageStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    pending_input_tokens: usize,
    expected_request: &ChatRequest,
) -> Result<Option<PlannedCompaction>> {
    if deps.is_cancelled() {
        return Ok(None);
    }
    let Some(cfg) = deps.compaction_config else {
        return Ok(None);
    };
    let Some(provider_arc) = deps.compaction_provider else {
        log::debug!(
            "plan_pre_call_compaction: compaction_config set but compaction_provider \
             missing on RootTurnDeps; skipping pre-call check"
        );
        return Ok(None);
    };

    let history = staged_messages
        .get_history(thread_id)
        .await
        .context("read staged history for compaction-threshold check")?;
    if !cfg.auto_compact || history.len() < cfg.min_messages_for_compaction {
        return Ok(None);
    }

    let (threshold_tokens, threshold_source) =
        serving_threshold_tokens(cfg, main_provider, expected_request.max_tokens);
    let compactor = build_compactor(
        deps,
        provider_arc,
        cfg,
        threshold_tokens,
        CompactionPurpose::PreSpawn,
    );
    let (trigger_tokens, trigger_source) =
        if cfg.uses_prune_first_engine() || cfg.uses_snapcompact_engine() {
            let measured = measured_occupancy(
                deps,
                main_provider,
                &compactor,
                &history,
                thread_id,
                expected_request,
            )
            .await?;
            trigger_tokens(measured, || compactor.estimate_tokens(&history))
        } else {
            (compactor.estimate_tokens(&history), "legacy_estimated")
        };
    let trigger_tokens = trigger_tokens.saturating_add(pending_input_tokens);
    if trigger_tokens <= threshold_tokens {
        return Ok(None);
    }

    log::info!(
        "Pre-call auto-compaction planned (thread={thread_id}, message_count={}, \
         trigger_tokens={trigger_tokens}, trigger_source={trigger_source}, \
         pending_input_tokens={pending_input_tokens}, \
         threshold_tokens={threshold_tokens}, threshold_source={threshold_source}, \
         provider={}, model={}, engine={:?}, purpose=pre_spawn)",
        history.len(),
        main_provider.provider(),
        main_provider.model(),
        cfg.engine,
    );
    Ok(Some(PlannedCompaction { compactor, history }))
}

/// Anchor tokens plus the estimate of the staged suffix the anchor did not
/// cover, or `None` when no attempt row qualifies.
async fn measured_occupancy(
    deps: &RootTurnDeps<'_>,
    main_provider: &dyn LlmProvider,
    compactor: &LlmContextCompactor<dyn LlmProvider>,
    history: &[agent_sdk_foundation::llm::Message],
    thread_id: &agent_sdk_foundation::ThreadId,
    expected_request: &ChatRequest,
) -> Result<Option<usize>> {
    let fingerprint = match request_envelope_fingerprint(
        main_provider.provider(),
        main_provider.model(),
        expected_request,
    ) {
        Ok(fingerprint) => fingerprint,
        Err(error) => {
            log::warn!("thread {thread_id}: {error:#}; estimating the full request instead");
            return Ok(None);
        }
    };
    let current_history_generation = deps
        .message_store
        .get(thread_id)
        .await
        .context("read projection for measured-trigger lineage")?
        .map_or(0, |projection| projection.history_generation);
    let requirements = AnchorRequirements {
        current_history_generation,
        history_len: history.len(),
        expected_request_fingerprint: &fingerprint,
        required_served_identity: (!main_provider.has_stable_request_route()).then(|| {
            ServedIdentity {
                route: main_provider.route(),
                model: main_provider.model(),
            }
        }),
    };
    Ok(latest_measured_anchor(deps, thread_id, &requirements)
        .await?
        .map(|anchor| {
            let appended = history
                .get(anchor.request_message_count..)
                .map_or(0, |suffix| compactor.estimate_tokens(suffix));
            anchor.tokens.saturating_add(appended)
        }))
}

/// Execute a compaction planned by [`plan_pre_call_compaction`]. The caller
/// owns the dedicated audit row and closes it with the outcome's exact
/// `llm_usage`, including completed-but-unapplied skips.
///
/// # Errors
///
/// Returns an error if the compactor's LLM call fails, or if either the
/// durable projection rewrite or the in-memory staged-buffer rewrite
/// fails.
pub(crate) async fn run_planned_compaction(
    deps: &RootTurnDeps<'_>,
    staged_messages: &StagedMessageStore,
    planned: PlannedCompaction,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
) -> CompactionResult<CompactionOutcome> {
    let PlannedCompaction { compactor, history } = planned;
    apply_compaction(
        deps,
        staged_messages,
        &compactor,
        history,
        thread_id,
        now,
        CompactionTrigger::Threshold,
    )
    .await
    .map_err(|failure| failure.with_context("pre-call compaction"))
}

#[cfg(test)]
pub(crate) async fn maybe_compact_staged_history(
    deps: &RootTurnDeps<'_>,
    main_provider: &dyn LlmProvider,
    staged_messages: &StagedMessageStore,
    thread_id: &agent_sdk_foundation::ThreadId,
    pending_input_tokens: usize,
    expected_request: &ChatRequest,
    now: OffsetDateTime,
) -> CompactionResult<CompactionOutcome> {
    let planned = plan_pre_call_compaction(
        deps,
        main_provider,
        staged_messages,
        thread_id,
        pending_input_tokens,
        expected_request,
    )
    .await?;
    match planned {
        None => Ok(CompactionOutcome::not_run()),
        Some(planned) => {
            run_planned_compaction(deps, staged_messages, planned, thread_id, now).await
        }
    }
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
    main_provider: &dyn LlmProvider,
    requested_max_tokens: u32,
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

    let (threshold_tokens, _) = serving_threshold_tokens(cfg, main_provider, requested_max_tokens);
    let compactor = build_compactor(
        deps,
        provider_arc,
        cfg,
        threshold_tokens,
        CompactionPurpose::Overflow,
    );

    log::warn!(
        "Provider rejected turn with prompt-too-long; attempting emergency \
         compaction (thread={thread_id}, message_count={}, engine={:?}, purpose=overflow)",
        history.len(),
        cfg.engine,
    );
    apply_compaction(
        deps,
        staged_messages,
        &compactor,
        history,
        thread_id,
        now,
        CompactionTrigger::Overflow,
    )
    .await
    .map_err(|failure| failure.with_context("overflow recovery compaction"))
}

/// True when a failed `append_compaction` was refused for breaking the
/// provider `tool_use`/`tool_result` sequence (ENG-9651) — the signal to skip
/// the compaction rather than fail the turn.
fn is_tool_sequence_refusal(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<crate::journal::message::MessageProjectionError>()
        .is_some_and(|projection_error| {
            matches!(
                projection_error,
                crate::journal::message::MessageProjectionError::CompactionBreaksToolSequence { .. }
            )
        })
}

/// True when the compactor produced no usable result and the turn should
/// proceed uncompacted rather than fail: either no token savings
/// (`NoProgressCompaction`) or the summarizer returned no text
/// (`SummarizationEmpty`, e.g. an adaptive-thinking model whose response
/// carried only thinking blocks). Both are benign; the provider call goes
/// out with the uncompacted history.
fn is_no_progress(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<agent_sdk::context::NoProgressCompaction>()
        .is_some()
        || error
            .downcast_ref::<agent_sdk::context::SummarizationEmpty>()
            .is_some()
}

/// Run the compactor and apply its result: append the durable compaction,
/// swap the staged view, and emit `ContextCompacted`.
///
/// Snapcompact artifacts a result already published are reclaimed when the
/// result drops before `append_compaction` succeeds (cancellation,
/// tool-sequence refusal, append failure); its rollback token is committed
/// right after the append. The cancel arm drops the compaction future while
/// the Snapcompact `spawn_blocking` worker keeps running, so callers wire the
/// same cancel token into the compactor as that worker's publish fence.
async fn apply_compaction(
    deps: &RootTurnDeps<'_>,
    staged_messages: &StagedMessageStore,
    compactor: &LlmContextCompactor<dyn LlmProvider>,
    history: Vec<agent_sdk_foundation::llm::Message>,
    thread_id: &agent_sdk_foundation::ThreadId,
    now: OffsetDateTime,
    trigger: CompactionTrigger,
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
            // A no-progress compaction is not an error: the assembled view
            // would be as large as the source, so there is nothing useful
            // to compact. Skip and let the provider call go out with the
            // uncompacted history (matches the tool-sequence refusal).
            if is_no_progress(&failure.error) {
                log::info!("thread {thread_id}: compaction made no progress; skipping compaction");
                return Ok(SummarizerSpend {
                    llm_usage: failure.llm_usage,
                    response_model: failure.response_model,
                    served_route: failure.served_route,
                }
                .outcome(false));
            }
            return Err(CompactionFailure::from(failure)
                .with_context("compactor.compact_history_with_usage failed"));
        }
    };
    let spend = SummarizerSpend {
        llm_usage: result.llm_usage.clone(),
        response_model: result.response_model.clone(),
        served_route: result.served_route.clone(),
    };

    // Provider usage becomes live-accounted before cancellation or any
    // durability decision. If cancellation completed in the same scheduler
    // turn as the response, the attempt still owns these billed tokens while
    // projection, staged history, and events remain untouched.
    if deps.is_cancelled() {
        return Ok(spend.outcome(false));
    }

    // A boundary that would cut a tool_use/tool_result pair is refused by
    // the storage layer (ENG-9651). That refusal must never fail the turn:
    // the compaction is skipped and the provider call goes out with the
    // uncompacted history.
    let appended = deps
        .message_store
        .append_compaction(
            thread_id,
            result.messages.clone(),
            result.original_count,
            result.retained_count,
            now,
        )
        .await;
    if let Err(error) = appended {
        if is_tool_sequence_refusal(&error) {
            log::warn!(
                "thread {thread_id}: compaction result broke the provider tool sequence; skipping compaction"
            );
            return Ok(spend.outcome(false));
        }
        return Err(spend.failure(error.context("append durable compaction entry")));
    }
    result.published_artifacts.commit();

    staged_messages
        .replace_history(thread_id, result.messages.clone())
        .await
        .context("replace staged buffer history")
        .map_err(|error| spend.failure(error))?;
    staged_messages.bump_history_generation();

    let event = AgentEvent::context_compacted(
        result.original_count,
        result.new_count,
        result.original_tokens,
        result.new_tokens,
        trigger,
        deps.compaction_config.map(|cfg| cfg.engine),
    );
    let committed = deps
        .event_repo
        .commit_event(thread_id, event, now)
        .await
        .context("commit ContextCompacted event")
        .map_err(|error| spend.failure(error))?;
    deps.event_notifier.notify(std::slice::from_ref(&committed));

    log::info!(
        "Auto-compaction complete (thread={thread_id}, \
         original_count={}, new_count={}, original_tokens={}, new_tokens={})",
        result.original_count,
        result.new_count,
        result.original_tokens,
        result.new_tokens,
    );
    Ok(spend.outcome(true))
}

/// Billed summarizer usage and identity of a compaction that completed.
struct SummarizerSpend {
    llm_usage: TokenUsage,
    response_model: Option<String>,
    served_route: Option<String>,
}

impl SummarizerSpend {
    fn outcome(self, applied: bool) -> CompactionOutcome {
        CompactionOutcome {
            completed: true,
            applied,
            llm_usage: self.llm_usage,
            response_model: self.response_model,
            served_route: self.served_route,
        }
    }

    fn failure(&self, error: anyhow::Error) -> CompactionFailure {
        CompactionFailure {
            error,
            llm_usage: self.llm_usage.clone(),
            response_model: self.response_model.clone(),
            served_route: self.served_route.clone(),
        }
    }
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
        let (tokens, source) = trigger_tokens(Some(88_000), || 1_200_000);
        assert_eq!(tokens, 88_000);
        assert_eq!(source, "measured");
        assert!(tokens <= 1_000_000);
    }

    #[test]
    fn estimator_is_used_only_before_measured_usage_exists() {
        let (tokens, source) = trigger_tokens(None, || 120_000);
        assert_eq!(tokens, 120_000);
        assert_eq!(source, "estimated_fallback");
    }

    #[test]
    fn full_history_estimate_is_lazy_when_a_measured_anchor_exists() {
        let (tokens, _) = trigger_tokens(Some(88_000), || {
            panic!("estimated_fallback must not be evaluated when measured is Some")
        });
        assert_eq!(tokens, 88_000);
    }

    #[test]
    fn no_progress_compaction_is_classified_as_skippable() {
        let no_progress = anyhow::Error::new(agent_sdk::context::NoProgressCompaction {
            new_tokens: 20_066,
            original_tokens: 20_052,
        });
        assert!(is_no_progress(&no_progress));

        // A real error is not a no-progress skip.
        let genuine = anyhow::anyhow!("provider stream failed");
        assert!(!is_no_progress(&genuine));

        // The marker survives anyhow context wrapping (downcast walks the chain).
        let wrapped = no_progress.context("pre-call compaction");
        assert!(is_no_progress(&wrapped));
    }

    fn registered(provider: &str, model: &str) -> Option<&'static ModelCapabilities> {
        agent_sdk_providers::get_model_capabilities(provider, model)
    }

    const FABLE_REQUESTED_MAX_TOKENS: u32 = 32_000;

    #[test]
    fn fixed_threshold_is_authoritative_when_opt_in_is_disabled() {
        let cfg = CompactionConfig::default().with_threshold_tokens(80_000);
        let (threshold, source) = effective_threshold_tokens(
            &cfg,
            registered("anthropic", "claude-fable-5"),
            FABLE_REQUESTED_MAX_TOKENS,
        );
        assert_eq!(threshold, 80_000);
        assert_eq!(source, "fixed");
    }

    #[test]
    fn model_aware_threshold_derives_window_minus_registered_output_cap() {
        let cfg = CompactionConfig::default().with_model_aware_threshold(true);
        let (threshold, source) = effective_threshold_tokens(
            &cfg,
            registered("anthropic", "claude-fable-5"),
            FABLE_REQUESTED_MAX_TOKENS,
        );
        assert_eq!(threshold, 1_000_000 - 128_000);
        assert_eq!(source, "model_aware");
    }

    #[test]
    fn explicit_reserve_takes_precedence_over_registered_output_cap() {
        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_model_context_reserve_tokens(50_000);
        let (threshold, source) = effective_threshold_tokens(
            &cfg,
            registered("anthropic", "claude-fable-5"),
            FABLE_REQUESTED_MAX_TOKENS,
        );
        assert_eq!(threshold, 1_000_000 - 50_000);
        assert_eq!(source, "model_aware");
    }

    #[test]
    fn reserve_never_drops_below_the_requested_max_tokens() {
        let caps = registered("anthropic", "claude-fable-5");
        let cfg = CompactionConfig::default().with_model_aware_threshold(true);
        assert_eq!(
            effective_threshold_tokens(&cfg, caps, 200_000),
            (1_000_000 - 200_000, "model_aware"),
        );

        let small_reserve = cfg.clone().with_model_context_reserve_tokens(10_000);
        assert_eq!(
            effective_threshold_tokens(&small_reserve, caps, 200_000),
            (1_000_000 - 200_000, "model_aware"),
        );

        let unknown_model_window = cfg.with_model_context_window_tokens(300_000);
        assert_eq!(
            effective_threshold_tokens(&unknown_model_window, None, 64_000),
            (300_000 - 64_000, "model_aware"),
        );
    }

    #[test]
    fn unset_requested_max_tokens_reserves_the_registered_output_cap() {
        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_model_context_reserve_tokens(10_000);
        assert_eq!(
            effective_threshold_tokens(&cfg, registered("anthropic", "claude-fable-5"), 0),
            (1_000_000 - 128_000, "model_aware"),
        );
    }

    #[test]
    fn unknown_model_falls_back_to_fixed_threshold_exactly() {
        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_threshold_tokens(77_000);
        let (threshold, source) =
            effective_threshold_tokens(&cfg, registered("mock", "mock-model"), 4_096);
        assert_eq!(threshold, 77_000);
        assert_eq!(source, "fixed_unknown_model");
    }

    #[test]
    fn explicit_context_window_overrides_registry_and_covers_unknown_models() {
        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_model_context_window_tokens(200_000);
        let (threshold, source) =
            effective_threshold_tokens(&cfg, registered("mock", "mock-model"), 4_096);
        assert_eq!(threshold, 200_000 - DEFAULT_MODEL_CONTEXT_RESERVE_TOKENS);
        assert_eq!(source, "model_aware");

        let (threshold, source) = effective_threshold_tokens(
            &cfg,
            registered("anthropic", "claude-fable-5"),
            FABLE_REQUESTED_MAX_TOKENS,
        );
        assert_eq!(threshold, 200_000 - 200_000 / 4);
        assert_eq!(source, "model_aware");
        let (threshold, _) =
            effective_threshold_tokens(&cfg, registered("anthropic", "claude-fable-5"), 128_000);
        assert_eq!(threshold, 200_000 - 128_000);
    }

    #[test]
    fn invalid_model_aware_window_falls_back_to_fixed_threshold() {
        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_threshold_tokens(77_000)
            .with_model_context_window_tokens(50_000)
            .with_model_context_reserve_tokens(50_000);
        let (threshold, source) =
            effective_threshold_tokens(&cfg, registered("mock", "mock-model"), 4_096);
        assert_eq!(threshold, 77_000);
        assert_eq!(source, "fixed_invalid_model_window");
    }

    #[test]
    fn model_aware_threshold_resolves_per_serving_route() {
        use agent_sdk_providers::{
            AnthropicProvider, GeminiProvider, OpenAICodexResponsesProvider, OpenAIProvider,
            OpenAIResponsesProvider,
        };

        const REQUESTED: u32 = 32_000;
        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_threshold_tokens(80_000);
        let gpt_threshold = 1_050_000 - 128_000;
        let kimi_threshold = 1_048_576 - DEFAULT_MODEL_CONTEXT_RESERVE_TOKENS.max(32_000);
        let routes: Vec<(&str, Box<dyn LlmProvider>, usize)> = vec![
            (
                "anthropic",
                Box::new(AnthropicProvider::new("key", "claude-fable-5")),
                1_000_000 - 128_000,
            ),
            (
                "openai",
                Box::new(OpenAIProvider::new("key", "gpt-5.6")),
                gpt_threshold,
            ),
            (
                "openai-responses",
                Box::new(OpenAIResponsesProvider::new("key".into(), "gpt-5.6".into())),
                gpt_threshold,
            ),
            (
                "openai-codex",
                Box::new(OpenAICodexResponsesProvider::new(
                    "key".into(),
                    "gpt-5.6".into(),
                )),
                gpt_threshold,
            ),
            (
                "gemini",
                Box::new(GeminiProvider::new("key", "gemini-3.1-pro")),
                1_048_576 - 65_536,
            ),
            (
                "openrouter",
                Box::new(OpenAIProvider::with_base_url(
                    "key",
                    "moonshotai/kimi-k3",
                    "https://openrouter.ai/api/v1",
                )),
                kimi_threshold,
            ),
            (
                "baseten",
                Box::new(OpenAIProvider::with_base_url(
                    "key",
                    "moonshotai/Kimi-K3",
                    "https://inference.baseten.co/v1",
                )),
                kimi_threshold,
            ),
            (
                "fireworks",
                Box::new(OpenAIProvider::with_base_url(
                    "key",
                    "accounts/fireworks/models/kimi-k3",
                    "https://api.fireworks.ai/inference/v1",
                )),
                kimi_threshold,
            ),
        ];
        for (route, provider, expected) in routes {
            assert_eq!(provider.route(), route);
            assert_eq!(
                serving_threshold_tokens(&cfg, provider.as_ref(), REQUESTED),
                (expected, "model_aware"),
                "route {route}",
            );
        }
    }

    struct RouteStub {
        provider: &'static str,
        model: &'static str,
        stable: bool,
    }

    #[async_trait::async_trait]
    impl LlmProvider for RouteStub {
        async fn chat(
            &self,
            _request: ChatRequest,
        ) -> Result<agent_sdk_foundation::llm::ChatOutcome> {
            Ok(agent_sdk_foundation::llm::ChatOutcome::ServerError(
                "not dispatched in tests".to_owned(),
            ))
        }

        fn model(&self) -> &str {
            self.model
        }

        fn provider(&self) -> &'static str {
            self.provider
        }

        fn has_stable_request_route(&self) -> bool {
            self.stable
        }
    }

    struct DelegatingGatewayStub {
        inner: RouteStub,
    }

    #[async_trait::async_trait]
    impl LlmProvider for DelegatingGatewayStub {
        async fn chat(
            &self,
            _request: ChatRequest,
        ) -> Result<agent_sdk_foundation::llm::ChatOutcome> {
            Ok(agent_sdk_foundation::llm::ChatOutcome::ServerError(
                "not dispatched in tests".to_owned(),
            ))
        }

        fn model(&self) -> &'static str {
            "gateway-alias"
        }

        fn provider(&self) -> &'static str {
            "test-gateway"
        }

        fn capabilities(&self) -> Option<&'static ModelCapabilities> {
            self.inner.capabilities()
        }
    }

    #[test]
    fn vertex_routes_resolve_their_model_family() {
        let cfg = CompactionConfig::default().with_model_aware_threshold(true);
        let vertex = RouteStub {
            provider: "vertex",
            model: "claude-fable-5",
            stable: true,
        };
        assert_eq!(
            serving_threshold_tokens(&cfg, &vertex, FABLE_REQUESTED_MAX_TOKENS),
            (1_000_000 - 128_000, "model_aware"),
        );
    }

    #[test]
    fn unstable_routes_contribute_no_capabilities() {
        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_threshold_tokens(80_000);
        let router = RouteStub {
            provider: "anthropic",
            model: "claude-fable-5",
            stable: false,
        };
        assert_eq!(
            serving_threshold_tokens(&cfg, &router, FABLE_REQUESTED_MAX_TOKENS),
            (80_000, "fixed_unknown_model"),
        );
    }

    #[test]
    fn gateway_delegated_capabilities_drive_the_threshold() {
        let cfg = CompactionConfig::default().with_model_aware_threshold(true);
        let gateway = DelegatingGatewayStub {
            inner: RouteStub {
                provider: "openai",
                model: "moonshotai/kimi-k3",
                stable: true,
            },
        };
        assert_eq!(
            serving_threshold_tokens(&cfg, &gateway, 4_096),
            (
                1_048_576 - DEFAULT_MODEL_CONTEXT_RESERVE_TOKENS,
                "model_aware"
            ),
        );
        let bare = RouteStub {
            provider: "test-gateway",
            model: "gateway-alias",
            stable: true,
        };
        assert_eq!(
            serving_threshold_tokens(&cfg, &bare, 4_096),
            (cfg.threshold_tokens, "fixed_unknown_model"),
        );
    }

    const ROOT_REQUESTED_MAX_TOKENS: u32 = 128_000;

    fn openrouter(model: &str) -> agent_sdk_providers::OpenAIProvider {
        agent_sdk_providers::OpenAIProvider::with_base_url(
            "key",
            model,
            "https://openrouter.ai/api/v1",
        )
    }

    /// The dispatched `max_tokens` and the threshold the pre-call trigger
    /// derives from that same request.
    fn worker_budget(
        cfg: &CompactionConfig,
        provider: &dyn LlmProvider,
        requested: u32,
    ) -> (u32, (usize, &'static str)) {
        thinking_worker_budget(cfg, provider, requested, None)
    }

    fn thinking_worker_budget(
        cfg: &CompactionConfig,
        provider: &dyn LlmProvider,
        requested: u32,
        thinking: Option<&ThinkingConfig>,
    ) -> (u32, (usize, &'static str)) {
        let capabilities = serving_capabilities(provider);
        let max_tokens = serving_max_tokens(
            requested,
            capabilities,
            OutputSizing::for_request(
                model_aware_trigger(Some(cfg), true),
                capabilities,
                thinking,
                false,
            ),
        );
        (
            max_tokens,
            serving_threshold_tokens(cfg, provider, max_tokens),
        )
    }

    fn model_aware() -> CompactionConfig {
        CompactionConfig::default().with_model_aware_threshold(true)
    }

    /// `max_tokens` for a request under a model-aware trigger.
    fn sized(
        requested: u32,
        capabilities: Option<&ModelCapabilities>,
        thinking: Option<&ThinkingConfig>,
    ) -> u32 {
        serving_max_tokens(
            requested,
            capabilities,
            OutputSizing::for_request(
                model_aware_trigger(Some(&model_aware()), true),
                capabilities,
                thinking,
                false,
            ),
        )
    }

    #[test]
    fn inherited_root_output_budget_is_sized_to_the_serving_model() {
        use agent_sdk_providers::{AnthropicProvider, OpenAICodexResponsesProvider};

        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_threshold_tokens(80_000);
        let routes: Vec<(&str, Box<dyn LlmProvider>, u32, usize)> = vec![
            (
                "claude-fable-5-1",
                Box::new(AnthropicProvider::new("key", "claude-fable-5-1")),
                128_000,
                1_000_000 - 128_000,
            ),
            (
                "claude-opus-5-5",
                Box::new(AnthropicProvider::new("key", "claude-opus-5-5")),
                128_000,
                1_000_000 - 128_000,
            ),
            (
                "gpt-6.1-sol",
                Box::new(OpenAICodexResponsesProvider::new(
                    "key".into(),
                    "gpt-6.1-sol".into(),
                )),
                128_000,
                1_050_000 - 128_000,
            ),
            (
                "z-ai/glm-5.1",
                Box::new(openrouter("z-ai/glm-5.1")),
                202_752 / 2,
                202_752 - 202_752 / 2,
            ),
            (
                "moonshotai/kimi-k3",
                Box::new(openrouter("moonshotai/kimi-k3")),
                128_000,
                1_048_576 - 128_000,
            ),
            (
                "accounts/fireworks/models/kimi-k3",
                Box::new(agent_sdk_providers::OpenAIProvider::with_base_url(
                    "key",
                    "accounts/fireworks/models/kimi-k3",
                    "https://api.fireworks.ai/inference/v1",
                )),
                128_000,
                1_048_576 - 128_000,
            ),
        ];
        for (model, provider, expected_max_tokens, expected_threshold) in routes {
            assert_eq!(
                worker_budget(&cfg, provider.as_ref(), ROOT_REQUESTED_MAX_TOKENS),
                (expected_max_tokens, (expected_threshold, "model_aware")),
                "{model}",
            );
        }
        let unknown = RouteStub {
            provider: "mock",
            model: "mock-model",
            stable: true,
        };
        assert_eq!(
            worker_budget(&cfg, &unknown, ROOT_REQUESTED_MAX_TOKENS),
            (128_000, (80_000, "fixed_unknown_model")),
        );
    }

    #[test]
    fn registered_output_cap_reserve_is_sized_like_the_request() {
        let cfg = CompactionConfig::default().with_model_aware_threshold(true);
        let glm = registered("openai", "z-ai/glm-5.1");
        assert_eq!(
            effective_threshold_tokens(&cfg, glm, 32_000),
            (202_752 - 202_752 / 4, "model_aware"),
        );
        assert_eq!(
            effective_threshold_tokens(&cfg, glm, 0),
            (202_752 - 202_752 / 4, "model_aware"),
        );
    }

    #[test]
    fn requests_above_the_model_max_output_are_clamped_to_it() {
        let fable = registered("anthropic", "claude-fable-5-1");
        let flash = registered("gemini", "gemini-2.0-flash");
        assert_eq!(
            serving_max_tokens(200_000, fable, OutputSizing::ModelMax),
            128_000
        );
        assert_eq!(sized(200_000, fable, None), 128_000);
        assert_eq!(
            serving_max_tokens(128_000, flash, OutputSizing::ModelMax),
            8_192
        );
        assert_eq!(sized(128_000, flash, None), 8_192);
        assert_eq!(
            sized(200_000, fable, Some(&ThinkingConfig::new(150_000))),
            128_000
        );
    }

    #[test]
    fn unknown_models_keep_the_requested_output_budget() {
        assert_eq!(
            serving_max_tokens(128_000, None, OutputSizing::ModelMax),
            128_000
        );
        assert_eq!(sized(128_000, None, None), 128_000);
        assert_eq!(
            sized(128_000, registered("mock", "mock-model"), None),
            128_000
        );
    }

    #[test]
    fn window_share_needs_a_model_aware_trigger_that_can_run() {
        let cfg = model_aware();
        assert!(model_aware_trigger(Some(&cfg), true).is_some());
        assert!(model_aware_trigger(None, true).is_none());
        assert!(model_aware_trigger(Some(&CompactionConfig::default()), true).is_none());
        assert!(
            model_aware_trigger(Some(&cfg), false).is_none(),
            "no compaction provider can run the trigger",
        );
        assert!(
            model_aware_trigger(Some(&model_aware().with_auto_compact(false)), true).is_none(),
            "automatic compaction is off",
        );

        let sonnet = registered("anthropic", "claude-sonnet-4-5-20250929");
        assert_eq!(
            OutputSizing::for_request(None, sonnet, None, false),
            OutputSizing::ModelMax,
        );
        let trigger = model_aware_trigger(Some(&cfg), true);
        assert_eq!(
            OutputSizing::for_request(trigger, None, None, false),
            OutputSizing::ModelMax,
            "no window to share",
        );
        for (thinking, reasoning) in [
            (None, ReasoningAllowance::None),
            (
                Some(ThinkingConfig::new(55_000)),
                ReasoningAllowance::Budget(55_000),
            ),
            (
                Some(ThinkingConfig::adaptive()),
                ReasoningAllowance::Unbudgeted,
            ),
            (
                Some(ThinkingConfig::default_with_effort(
                    agent_sdk_foundation::llm::Effort::High,
                )),
                ReasoningAllowance::Unbudgeted,
            ),
        ] {
            assert_eq!(
                OutputSizing::for_request(trigger, sonnet, thinking.as_ref(), false),
                OutputSizing::WindowShare {
                    window: 200_000,
                    reasoning,
                },
            );
        }
    }

    #[test]
    fn without_a_model_aware_threshold_only_the_model_max_bounds_output() {
        for (provider, model, requested, expected) in [
            ("anthropic", "claude-sonnet-4-5-20250929", 64_000, 64_000),
            ("anthropic", "claude-haiku-4-5-20251001", 64_000, 64_000),
            ("openai", "gpt-5", 128_000, 128_000),
            ("openai", "o3", 100_000, 100_000),
            ("openai", "deepseek/deepseek-v4-pro", 384_000, 384_000),
            ("openai", "minimax/minimax-m2.5", 128_000, 128_000),
            ("openai", "z-ai/glm-5.1", 128_000, 128_000),
            ("openai", "z-ai/glm-5.1", 100_000, 100_000),
            ("openai", "z-ai/glm-5.1", 200_000, 131_072),
        ] {
            assert_eq!(
                serving_max_tokens(
                    requested,
                    registered(provider, model),
                    OutputSizing::ModelMax
                ),
                expected,
                "{model} at {requested}",
            );
        }
        let glm = openrouter("z-ai/glm-5.1");
        assert_eq!(
            worker_budget(
                &CompactionConfig::default().with_threshold_tokens(80_000),
                &glm,
                ROOT_REQUESTED_MAX_TOKENS,
            ),
            (128_000, (80_000, "fixed")),
        );
    }

    #[test]
    fn window_share_bounds_output_only_where_it_binds() -> anyhow::Result<()> {
        let registered_kimi = registered("openai", "accounts/fireworks/models/kimi-k3")
            .context("fireworks kimi k3 row")?;
        assert_eq!(registered_kimi.max_output_tokens, None);
        assert_eq!(registered_kimi.default_reasoning, DefaultReasoning::On);
        assert_eq!(sized(600_000, Some(registered_kimi), None), 1_048_576 / 2);
        let kimi = registered_kimi.with_default_reasoning(DefaultReasoning::Off);
        assert_eq!(sized(128_000, Some(&kimi), None), 128_000);
        assert_eq!(sized(400_000, Some(&kimi), None), 1_048_576 / 4);

        let small_window = kimi.with_context_window(Some(100_000));
        assert_eq!(sized(128_000, Some(&small_window), None), 32_000);
        assert_eq!(sized(16_000, Some(&small_window), None), 16_000);
        Ok(())
    }

    #[test]
    fn reasoning_rides_on_top_of_the_window_share() {
        let budget = |tokens| Some(ThinkingConfig::new(tokens));
        let sonnet = registered("anthropic", "claude-sonnet-4-5-20250929");
        assert_eq!(sized(64_000, sonnet, None), 50_000);
        assert_eq!(sized(64_000, sonnet, budget(10_000).as_ref()), 60_000);
        assert_eq!(
            sized(64_000, sonnet, budget(49_999).as_ref()),
            64_000,
            "the model max, not the window share, bounds a budget near the share",
        );
        assert_eq!(sized(128_000, sonnet, budget(55_000).as_ref()), 64_000);

        let glm = registered("openai", "z-ai/glm-5.1");
        let budgeted = sized(128_000, glm, budget(51_000).as_ref());
        assert_eq!(budgeted, 202_752 / 4 + 51_000);
        assert_eq!(
            effective_threshold_tokens(&model_aware(), glm, budgeted),
            (202_752 - (202_752 / 4 + 51_000), "model_aware"),
        );

        let effort = Some(ThinkingConfig::default_with_effort(
            agent_sdk_foundation::llm::Effort::High,
        ));
        let o3 = registered("openai", "o3");
        assert_eq!(
            sized(100_000, o3, effort.as_ref()),
            100_000,
            "unbudgeted reasoning keeps a second share, here the whole model max",
        );
        assert_eq!(
            sized(100_000, o3, Some(&ThinkingConfig::adaptive())),
            100_000
        );
        assert_eq!(sized(128_000, glm, effort.as_ref()), 202_752 / 2);
    }

    #[test]
    fn models_that_reason_by_default_get_the_reasoning_share() {
        let override_cfg = model_aware().with_model_context_window_tokens(200_000);
        let override_sized = |model: &str| {
            let capabilities = registered("anthropic", model);
            serving_max_tokens(
                ROOT_REQUESTED_MAX_TOKENS,
                capabilities,
                OutputSizing::for_request(
                    model_aware_trigger(Some(&override_cfg), true),
                    capabilities,
                    None,
                    false,
                ),
            )
        };
        assert_eq!(sized(100_000, registered("openai", "o3"), None), 100_000);
        assert_eq!(override_sized("claude-opus-5-5"), 100_000);
        assert_eq!(override_sized("claude-opus-4-8"), 50_000);
        let glm = registered("openai", "z-ai/glm-5.1");
        let glm_request = sized(ROOT_REQUESTED_MAX_TOKENS, glm, None);
        assert_eq!(glm_request, 202_752 / 2);
        assert_eq!(
            effective_threshold_tokens(&model_aware(), glm, glm_request),
            (202_752 - 202_752 / 2, "model_aware"),
        );
        assert_eq!(
            sized(
                64_000,
                registered("anthropic", "claude-sonnet-4-5-20250929"),
                None
            ),
            50_000,
            "a model that thinks only when asked gets the visible share alone",
        );
    }

    #[test]
    fn window_override_sizes_output_like_the_threshold() {
        let cfg = model_aware()
            .with_model_context_window_tokens(200_000)
            .with_threshold_tokens(80_000);
        let opus = agent_sdk_providers::AnthropicProvider::new("key", "claude-opus-5-5");
        let opus_48 = agent_sdk_providers::AnthropicProvider::new("key", "claude-opus-4-8");
        assert_eq!(
            worker_budget(&cfg, &opus_48, ROOT_REQUESTED_MAX_TOKENS),
            (50_000, (150_000, "model_aware")),
        );
        assert_eq!(
            worker_budget(&cfg, &opus, ROOT_REQUESTED_MAX_TOKENS),
            (100_000, (100_000, "model_aware")),
        );
        assert_eq!(
            thinking_worker_budget(
                &cfg,
                &opus,
                ROOT_REQUESTED_MAX_TOKENS,
                Some(&ThinkingConfig::adaptive()),
            ),
            (100_000, (100_000, "model_aware")),
        );
        for (route, stable) in [("unknown", true), ("unstable", false)] {
            let provider = RouteStub {
                provider: "mock",
                model: "mock-model",
                stable,
            };
            assert_eq!(
                worker_budget(&cfg, &provider, ROOT_REQUESTED_MAX_TOKENS),
                (128_000, (200_000 - 128_000, "model_aware")),
                "{route}: no capability row, so the request's reasoning is unknown",
            );
        }
        assert_eq!(
            worker_budget(
                &model_aware().with_threshold_tokens(80_000),
                &opus,
                ROOT_REQUESTED_MAX_TOKENS,
            ),
            (128_000, (872_000, "model_aware")),
        );
    }

    #[test]
    fn unstable_routes_trust_no_output_cap() {
        let router = RouteStub {
            provider: "openai",
            model: "z-ai/glm-5.1",
            stable: false,
        };
        assert!(serving_capabilities(&router).is_none());
        assert_eq!(sized(128_000, serving_capabilities(&router), None), 128_000);
    }

    #[test]
    fn in_tree_wrappers_size_output_from_the_same_row_as_the_threshold() {
        use agent_sdk_providers::{FallbackProvider, ModelRouter, RefreshingProvider};

        let cfg = CompactionConfig::default()
            .with_model_aware_threshold(true)
            .with_threshold_tokens(80_000);
        let glm = openrouter("z-ai/glm-5.1");
        let template = glm.clone();
        let refreshing = RefreshingProvider::new(glm.clone(), move || {
            let provider = template.clone();
            async move { Ok(provider) }
        });
        let failover = FallbackProvider::new(Arc::new(glm.clone()));
        let model_router = ModelRouter::new(
            openrouter("mock-classifier"),
            openrouter("mock-fast"),
            glm,
            openrouter("mock-advanced"),
        );
        assert_eq!(failover.model(), "z-ai/glm-5.1");
        assert_eq!(LlmProvider::model(&model_router), "z-ai/glm-5.1");
        let budget =
            |provider: &dyn LlmProvider| worker_budget(&cfg, provider, ROOT_REQUESTED_MAX_TOKENS);
        assert_eq!(
            budget(&refreshing),
            (202_752 / 2, (202_752 - 202_752 / 2, "model_aware")),
            "refreshing",
        );
        assert_eq!(
            budget(&failover),
            (ROOT_REQUESTED_MAX_TOKENS, (80_000, "fixed_unknown_model")),
            "failover",
        );
        assert_eq!(
            budget(&model_router),
            (ROOT_REQUESTED_MAX_TOKENS, (80_000, "fixed_unknown_model")),
            "router",
        );
    }

    fn anchor_time() -> anyhow::Result<OffsetDateTime> {
        OffsetDateTime::from_unix_timestamp(1_700_000_000).map_err(anyhow::Error::from)
    }

    fn mock_envelope() -> agent_sdk_foundation::llm::ChatRequest {
        agent_sdk_foundation::llm::ChatRequest::new("You are a helpful test assistant.", vec![])
    }

    fn mock_fingerprint() -> anyhow::Result<String> {
        request_envelope_fingerprint("mock", "mock-model", &mock_envelope())
    }

    fn stamped_attempt(
        opened_at: OffsetDateTime,
        stamps: &serde_json::Value,
    ) -> anyhow::Result<TurnAttempt> {
        use crate::journal::task::AgentTaskId;
        use crate::journal::turn_attempt::{CloseAttemptParams, OpenAttemptParams};
        use agent_sdk_foundation::audit::AuditProvenance;

        let mut response_blob = serde_json::json!({
            "model": "mock-model",
            "request_input_tokens": 500_000,
            "request_message_count": 7,
            "request_fingerprint": mock_fingerprint()?,
            "history_generation": 1,
        });
        if let Some(overrides) = stamps.as_object() {
            let blob = response_blob.as_object_mut().context("blob is an object")?;
            for (key, value) in overrides {
                if value.is_null() && key.starts_with('-') {
                    blob.remove(key.trim_start_matches('-'));
                } else {
                    blob.insert(key.clone(), value.clone());
                }
            }
        }
        let attempt = TurnAttempt::open(OpenAttemptParams {
            task_id: AgentTaskId::from_string("task_anchor"),
            attempt_number: 1,
            provenance: AuditProvenance::new("mock", "mock-model"),
            request_blob: serde_json::json!({ "user_prompt": "do the thing" }),
            now: opened_at,
            otel_trace_id: None,
            otel_span_id: None,
        });
        attempt
            .close(
                CloseAttemptParams {
                    response_blob,
                    response_id: None,
                    response_model: None,
                    stop_reason: None,
                    outcome: TurnAttemptOutcome::Success,
                    input_tokens: 1_000_000,
                    output_tokens: 100_000,
                    cached_input_tokens: 450_000,
                    cache_creation_input_tokens: 50_000,
                    route_provider: None,
                    thinking_mode: None,
                    thinking_budget_tokens: None,
                    thinking_effort: None,
                },
                opened_at + time::Duration::seconds(1),
            )
            .map_err(anyhow::Error::from)
    }

    fn requirements(fingerprint: &str) -> AnchorRequirements<'_> {
        AnchorRequirements {
            current_history_generation: 1,
            history_len: 24,
            expected_request_fingerprint: fingerprint,
            required_served_identity: None,
        }
    }

    #[test]
    fn unstable_route_anchors_only_on_a_confirmed_serving_identity() -> anyhow::Result<()> {
        let fingerprint = mock_fingerprint()?;
        let unstable = AnchorRequirements {
            required_served_identity: Some(ServedIdentity {
                route: "primary",
                model: "primary-model",
            }),
            ..requirements(&fingerprint)
        };
        let served_by_primary = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "served_route": "primary", "served_model": "primary-model" }),
        )?;
        assert!(attempt_measured_anchor(&served_by_primary, &unstable).is_some());
        let failed_over = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "served_route": "secondary", "served_model": "primary-model" }),
        )?;
        assert!(attempt_measured_anchor(&failed_over, &unstable).is_none());
        let other_model_same_route = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "served_route": "primary", "served_model": "other-model" }),
        )?;
        assert!(
            attempt_measured_anchor(&other_model_same_route, &unstable).is_none(),
            "a route can serve several models; another model's count never anchors",
        );
        let route_only = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "served_route": "primary" }),
        )?;
        assert!(
            attempt_measured_anchor(&route_only, &unstable).is_none(),
            "a row without a stream-confirmed model cannot prove which model it measured",
        );
        let unconfirmed = stamped_attempt(anchor_time()?, &serde_json::json!({}))?;
        assert!(
            attempt_measured_anchor(&unconfirmed, &unstable).is_none(),
            "a row without a stream-confirmed route cannot prove which backend it measured",
        );
        assert!(
            attempt_measured_anchor(&unconfirmed, &requirements(&fingerprint)).is_some(),
            "a stable route needs no serving confirmation",
        );
        Ok(())
    }

    #[test]
    fn measured_occupancy_is_the_trusted_physical_request_total() -> anyhow::Result<()> {
        let fingerprint = mock_fingerprint()?;
        let attempt = stamped_attempt(anchor_time()?, &serde_json::json!({}))?;
        let (_, anchor) = attempt_measured_anchor(&attempt, &requirements(&fingerprint))
            .context("expected an anchor")?;
        assert_eq!(anchor.tokens, 500_000);
        assert_eq!(anchor.request_message_count, 7);
        Ok(())
    }

    #[test]
    fn aggregate_billing_totals_cannot_anchor() -> anyhow::Result<()> {
        let fingerprint = mock_fingerprint()?;
        let missing = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "-request_input_tokens": null }),
        )?;
        assert!(attempt_measured_anchor(&missing, &requirements(&fingerprint)).is_none());

        let explicit_null = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "request_input_tokens": null }),
        )?;
        assert!(attempt_measured_anchor(&explicit_null, &requirements(&fingerprint)).is_none());

        let zero = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "request_input_tokens": 0 }),
        )?;
        assert!(attempt_measured_anchor(&zero, &requirements(&fingerprint)).is_none());
        Ok(())
    }

    #[test]
    fn compaction_audit_attempts_cannot_anchor() -> anyhow::Result<()> {
        let fingerprint = mock_fingerprint()?;
        for applied in [false, true] {
            let attempt = stamped_attempt(
                anchor_time()?,
                &serde_json::json!({ "operation": "context_compaction", "applied": applied }),
            )?;
            assert!(
                attempt_measured_anchor(&attempt, &requirements(&fingerprint)).is_none(),
                "applied={applied}"
            );
        }
        Ok(())
    }

    #[test]
    fn user_prompt_matching_compaction_sentinel_still_anchors() -> anyhow::Result<()> {
        use crate::journal::task::AgentTaskId;
        use crate::journal::turn_attempt::{CloseAttemptParams, OpenAttemptParams};
        use agent_sdk_foundation::audit::AuditProvenance;
        let fingerprint = mock_fingerprint()?;
        let attempt = TurnAttempt::open(OpenAttemptParams {
            task_id: AgentTaskId::from_string("task_anchor"),
            attempt_number: 1,
            provenance: AuditProvenance::new("mock", "mock-model"),
            request_blob: serde_json::json!({ "user_prompt": COMPACTION_AUDIT_PROMPT }),
            now: anchor_time()?,
            otel_trace_id: None,
            otel_span_id: None,
        })
        .close(
            CloseAttemptParams {
                response_blob: serde_json::json!({
                    "request_input_tokens": 480_000,
                    "request_message_count": 7,
                    "request_fingerprint": fingerprint,
                    "history_generation": 1,
                }),
                response_id: None,
                response_model: None,
                stop_reason: None,
                outcome: TurnAttemptOutcome::Success,
                input_tokens: 480_000,
                output_tokens: 2_000,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                route_provider: None,
                thinking_mode: None,
                thinking_budget_tokens: None,
                thinking_effort: None,
            },
            anchor_time()? + time::Duration::seconds(1),
        )
        .map_err(anyhow::Error::from)?;
        assert!(attempt_measured_anchor(&attempt, &requirements(&fingerprint)).is_some());
        Ok(())
    }

    #[test]
    fn stale_history_generation_cannot_anchor() -> anyhow::Result<()> {
        let fingerprint = mock_fingerprint()?;
        let attempt = stamped_attempt(anchor_time()?, &serde_json::json!({}))?;
        let stale = AnchorRequirements {
            current_history_generation: 2,
            ..requirements(&fingerprint)
        };
        assert!(attempt_measured_anchor(&attempt, &stale).is_none());
        let unstamped = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "-history_generation": null }),
        )?;
        assert!(attempt_measured_anchor(&unstamped, &requirements(&fingerprint)).is_none());
        Ok(())
    }

    #[test]
    fn envelope_fingerprint_mismatch_cannot_anchor() -> anyhow::Result<()> {
        let attempt = stamped_attempt(anchor_time()?, &serde_json::json!({}))?;

        let other_system = request_envelope_fingerprint(
            "mock",
            "mock-model",
            &agent_sdk_foundation::llm::ChatRequest::new("A different profile prompt.", vec![]),
        )?;
        assert!(attempt_measured_anchor(&attempt, &requirements(&other_system)).is_none());

        let other_model = request_envelope_fingerprint("mock", "other-model", &mock_envelope())?;
        assert!(attempt_measured_anchor(&attempt, &requirements(&other_model)).is_none());

        let mut schema_envelope = mock_envelope();
        schema_envelope.response_format = Some(agent_sdk_foundation::llm::ResponseFormat::new(
            "out",
            serde_json::json!({ "type": "object" }),
        ));
        let with_schema = request_envelope_fingerprint("mock", "mock-model", &schema_envelope)?;
        assert!(
            attempt_measured_anchor(&attempt, &requirements(&with_schema)).is_none(),
            "a structured-output schema is provider input and changes the envelope",
        );

        let matching = mock_fingerprint()?;
        assert!(attempt_measured_anchor(&attempt, &requirements(&matching)).is_some());
        Ok(())
    }

    #[test]
    fn missing_or_oversized_message_count_cannot_anchor() -> anyhow::Result<()> {
        let fingerprint = mock_fingerprint()?;
        let missing = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "-request_message_count": null }),
        )?;
        assert!(attempt_measured_anchor(&missing, &requirements(&fingerprint)).is_none());

        let oversized = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "request_message_count": 25 }),
        )?;
        let small_history = AnchorRequirements {
            history_len: 24,
            ..requirements(&fingerprint)
        };
        assert!(attempt_measured_anchor(&oversized, &small_history).is_none());

        let exact = stamped_attempt(
            anchor_time()?,
            &serde_json::json!({ "request_message_count": 24 }),
        )?;
        assert!(attempt_measured_anchor(&exact, &small_history).is_some());
        Ok(())
    }

    #[test]
    fn request_envelope_fingerprint_is_deterministic_and_sensitive() -> anyhow::Result<()> {
        assert_eq!(mock_fingerprint()?, mock_fingerprint()?);
        let base = mock_fingerprint()?;
        assert_eq!(base.len(), 64, "hex-encoded SHA-256");

        assert_ne!(
            base,
            request_envelope_fingerprint("other-provider", "mock-model", &mock_envelope())?,
        );
        let mut with_tools = mock_envelope();
        with_tools.tools = Some(vec![agent_sdk_foundation::llm::Tool {
            name: "lookup".into(),
            description: "lookup".into(),
            input_schema: serde_json::json!({ "type": "object" }),
            display_name: "lookup".into(),
            tier: agent_sdk_foundation::ToolTier::Observe,
        }]);
        assert_ne!(
            base,
            request_envelope_fingerprint("mock", "mock-model", &with_tools)?,
        );
        let mut with_messages = mock_envelope();
        with_messages.messages = vec![agent_sdk_foundation::llm::Message::user("hello")];
        assert_eq!(
            base,
            request_envelope_fingerprint("mock", "mock-model", &with_messages)?,
        );
        Ok(())
    }
}
