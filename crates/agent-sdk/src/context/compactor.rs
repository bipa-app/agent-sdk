//! Context compaction implementation.

use crate::hooks::{AgentHooks, DefaultHooks, RequestDecision, ResponseDecision};
use crate::llm::{
    ChatOutcome, ChatRequest, Content, ContentBlock, LlmProvider, Message, Role, StopReason,
};
use crate::types::TokenUsage;
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::fmt::Write;
use std::sync::Arc;

use super::config::CompactionConfig;
use super::estimator::TokenEstimator;

const SUMMARY_PREFIX: &str = "[Previous conversation summary]\n\n";
const COMPACTION_SYSTEM_PROMPT: &str = "You are a precise summarizer. Your task is to create concise but complete summaries of conversations, preserving all technical details needed to continue the work.";
const COMPACTION_SUMMARY_PROMPT_PREFIX: &str = "Summarize this conversation concisely, preserving:\n- Key decisions and conclusions reached\n- Important file paths, code changes, and technical details\n- Current task context and what has been accomplished\n- Any pending items, errors encountered, or next steps\n\nBe specific about technical details (file names, function names, error messages) as these\nare critical for continuing the work.\n\nConversation:\n";
const COMPACTION_SUMMARY_PROMPT_SUFFIX: &str =
    "Provide a concise summary (aim for 500-1000 words):";
const COMPACT_EMPTY_SUMMARY: &str = "No additional context was available to summarize; the previous messages were already compacted.";
const SUMMARY_ACKNOWLEDGMENT: &str =
    "I understand the context from the summary. Let me continue from where we left off.";
const MAX_TOOL_RESULT_CHARS: usize = 500;
const TRUNCATED_SUMMARY_MARKER: &str =
    "\n\n[summary truncated: exceeded the configured summary_max_tokens budget]";

/// Trait for context compaction strategies.
///
/// Implement this trait to provide custom compaction logic.
#[async_trait]
pub trait ContextCompactor: Send + Sync {
    /// Compact a list of messages into a summary.
    ///
    /// # Errors
    /// Returns an error if summarization fails.
    async fn compact(&self, messages: &[Message]) -> Result<String>;

    /// Estimate tokens for a message list.
    fn estimate_tokens(&self, messages: &[Message]) -> usize;

    /// Check if compaction is needed.
    fn needs_compaction(&self, messages: &[Message]) -> bool;

    /// Perform full compaction, returning new message history.
    ///
    /// # Errors
    /// Returns an error if compaction fails.
    async fn compact_history(&self, messages: Vec<Message>) -> Result<CompactionResult>;

    /// Like [`compact_history`](Self::compact_history), but a failure
    /// additionally reports the provider-billed usage of any summarization
    /// calls already made, so callers can account billed-but-wasted spend.
    ///
    /// The default delegates to `compact_history` and reports zero usage on
    /// failure — custom compactors that bill LLM calls should override this
    /// (best-effort: an un-overridden custom compactor under-reports failed
    /// attempts' usage, never over-reports).
    ///
    /// # Errors
    /// Returns [`FailedCompaction`] when compaction fails.
    async fn compact_history_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> Result<CompactionResult, FailedCompaction> {
        self.compact_history(messages)
            .await
            .map_err(|error| FailedCompaction {
                error,
                llm_usage: TokenUsage::default(),
            })
    }
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// The new compacted message history.
    pub messages: Vec<Message>,
    /// Number of messages before compaction.
    pub original_count: usize,
    /// Number of messages after compaction.
    pub new_count: usize,
    /// Estimated tokens before compaction.
    pub original_tokens: usize,
    /// Estimated tokens after compaction.
    pub new_tokens: usize,
    /// Provider-billed usage of the summarization LLM call(s) that produced
    /// this result (zero when compaction completed without an LLM call).
    ///
    /// Surfaced so the agent loop can fold compaction spend into the run's
    /// cumulative usage — otherwise summarization tokens would be invisible
    /// to `UsageLimits` budgets and under-reported on `Done`.
    pub llm_usage: TokenUsage,
}

/// A failed compaction attempt, carrying the provider-billed usage of any
/// summarization LLM calls that were already made before the failure.
///
/// Surfaced by [`ContextCompactor::compact_history_with_usage`] so the agent
/// loop can fold billed-but-wasted summarization spend into the run's
/// cumulative usage even when the history is left uncompacted (guardrail
/// block, truncation-retry error, `replace_history` failure).
#[derive(Debug)]
pub struct FailedCompaction {
    /// Why the compaction attempt failed.
    pub error: anyhow::Error,
    /// Usage billed by summarization calls made before the failure (zero
    /// when the failure preceded any LLM call).
    pub llm_usage: TokenUsage,
}

/// LLM-based context compactor.
///
/// Uses the LLM itself to summarize older messages into a compact form.
///
/// # Budgets
///
/// The compactor performs no budget evaluation of its own: it may issue up
/// to **two** summarization LLM calls per compaction (the second only when
/// the first summary was truncated, retried with a doubled token budget)
/// before the agent loop's next [`UsageLimits`](crate::types::UsageLimits)
/// boundary check runs. Every call's usage — including failed attempts — is
/// reported via [`CompactionResult::llm_usage`] /
/// [`FailedCompaction::llm_usage`] and folded by the loop immediately after
/// compaction, so the overshoot is bounded and consistent with the loop's
/// boundary-check semantics.
///
/// `P` is `?Sized` so callers can hold an `Arc<dyn LlmProvider>` —
/// useful when the provider is resolved dynamically per-thread (e.g.
/// inside `agent-server`'s daemon worker, where the same compactor
/// type wraps whichever concrete provider the host's resolver picks).
/// Concrete-type users (`Arc<AnthropicProvider>`, etc.) still work
/// unchanged.
pub struct LlmContextCompactor<P: LlmProvider + ?Sized, H: AgentHooks = DefaultHooks> {
    provider: Arc<P>,
    config: CompactionConfig,
    /// Guardrail hooks applied to the summarization LLM call. `None` (the
    /// default) skips the guardrails, preserving the historical behavior for
    /// direct constructions; the agent loop always attaches its run hooks so
    /// compaction cannot bypass `pre_llm_request` / `on_llm_response`.
    hooks: Option<Arc<H>>,
    system_prompt: String,
    summary_prompt_prefix: String,
    summary_prompt_suffix: String,
}

impl<P: LlmProvider + ?Sized> LlmContextCompactor<P> {
    /// Create a new LLM context compactor.
    #[must_use]
    pub fn new(provider: Arc<P>, config: CompactionConfig) -> Self {
        Self {
            provider,
            config,
            hooks: None,
            system_prompt: COMPACTION_SYSTEM_PROMPT.to_string(),
            summary_prompt_prefix: COMPACTION_SUMMARY_PROMPT_PREFIX.to_string(),
            summary_prompt_suffix: COMPACTION_SUMMARY_PROMPT_SUFFIX.to_string(),
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults(provider: Arc<P>) -> Self {
        Self::new(provider, CompactionConfig::default())
    }
}

impl<P: LlmProvider + ?Sized, H: AgentHooks> LlmContextCompactor<P, H> {
    /// Apply the run's guardrail hooks to every summarization LLM call.
    ///
    /// `pre_llm_request` runs before the call (`Proceed`/`Modify` apply;
    /// `Block` aborts the compaction attempt with an error) and
    /// `on_llm_response` runs on the produced summary (`Accept` applies;
    /// `Block` **and** `RetryWithFeedback` abort the compaction attempt —
    /// the compactor never retries a rejected summary, so a
    /// deterministically-rejecting hook cannot start a paid retry loop
    /// here). An aborted compaction surfaces as a `compact_history` error;
    /// the agent loop then continues with the uncompacted history
    /// (threshold trigger) or fails the recovery (overflow trigger).
    #[must_use]
    pub fn with_guardrail_hooks<H2: AgentHooks>(
        self,
        hooks: Arc<H2>,
    ) -> LlmContextCompactor<P, H2> {
        LlmContextCompactor {
            provider: self.provider,
            config: self.config,
            hooks: Some(hooks),
            system_prompt: self.system_prompt,
            summary_prompt_prefix: self.summary_prompt_prefix,
            summary_prompt_suffix: self.summary_prompt_suffix,
        }
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &CompactionConfig {
        &self.config
    }

    /// Override the prompts used for LLM-based summarization.
    #[must_use]
    pub fn with_prompts(
        mut self,
        system_prompt: impl Into<String>,
        summary_prompt_prefix: impl Into<String>,
        summary_prompt_suffix: impl Into<String>,
    ) -> Self {
        self.system_prompt = system_prompt.into();
        self.summary_prompt_prefix = summary_prompt_prefix.into();
        self.summary_prompt_suffix = summary_prompt_suffix.into();
        self
    }

    /// If `content` is a previously inserted compaction summary, return its
    /// text with the `SUMMARY_PREFIX` marker stripped; otherwise `None`.
    ///
    /// Used to carry a prior summary's prose forward into the next compaction
    /// instead of discarding it (which silently destroyed all pre-first-
    /// compaction context). The marker is still a content-prefix sentinel
    /// because `Message` lives in a foundation crate and cannot carry a
    /// structural flag from here; the compactor itself is the only writer of
    /// the prefix.
    fn extract_summary_text(content: &Content) -> Option<String> {
        match content {
            Content::Text(text) => text.strip_prefix(SUMMARY_PREFIX).map(str::to_string),
            Content::Blocks(blocks) => blocks.iter().find_map(|block| match block {
                ContentBlock::Text { text } => {
                    text.strip_prefix(SUMMARY_PREFIX).map(str::to_string)
                }
                _ => None,
            }),
        }
    }

    /// Return true when a message contains a tool-use block.
    fn has_tool_use(content: &Content) -> bool {
        matches!(
            content,
            Content::Blocks(blocks)
                if blocks
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        )
    }

    /// Return true when a message contains a tool-result block.
    fn has_tool_result(content: &Content) -> bool {
        matches!(
            content,
            Content::Blocks(blocks)
                if blocks
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
        )
    }

    /// Return true when history contains provider-owned state that must be
    /// replayed byte-for-byte on a follow-up request.
    ///
    /// A generic prose summary cannot stand in for this state: doing so would
    /// both discard the payload and change its position relative to the
    /// surrounding conversation. Until a provider-native compaction path can
    /// preserve it, the only safe generic behavior is to leave the history
    /// intact.
    fn has_opaque_reasoning(messages: &[Message]) -> bool {
        messages.iter().any(|message| {
            matches!(
                &message.content,
                Content::Blocks(blocks)
                    if blocks
                        .iter()
                        .any(|block| matches!(block, ContentBlock::OpaqueReasoning { .. }))
            )
        })
    }

    /// Shift split point backwards until a `tool_use`/`tool_result` pair is not
    /// split.
    ///
    /// Only the `assistant(tool_use)` -> `user(tool_result)` boundary is
    /// unsplittable: that is the single tool turn that must stay together for
    /// the wire payload to be valid. Splitting at a `user(tool_result)` ->
    /// `assistant(tool_use)` boundary is API-valid (the retained tail then
    /// begins with an `assistant` `tool_use` followed by its own result), so
    /// it is *not* treated as a pair. Treating it as a pair used to walk the
    /// split backward through an entire unbroken tool chain — the dominant
    /// shape of autonomous traces — defeating the retained-tail token cap and
    /// summarizing almost nothing.
    fn split_point_preserves_tool_pairs(messages: &[Message], mut split_point: usize) -> usize {
        while split_point > 0 && split_point < messages.len() {
            let prev = &messages[split_point - 1];
            let next = &messages[split_point];

            let crosses_tool_pair = prev.role == Role::Assistant
                && Self::has_tool_use(&prev.content)
                && next.role == Role::User
                && Self::has_tool_result(&next.content);

            if crosses_tool_pair {
                split_point -= 1;
                continue;
            }

            break;
        }

        split_point
    }

    /// Pick a split point that produces a self-consistent `to_keep`.
    ///
    /// `to_keep` is self-consistent (per Anthropic's API contract)
    /// when every `tool_result` block it contains references a
    /// `tool_use` block earlier in `to_keep`. The compactor inserts
    /// a synthetic `[summary, summary_ack]` prefix in front of
    /// `to_keep`, and that prefix has no `tool_use` blocks — so the
    /// only path to a valid wire payload is for `to_keep` itself to
    /// be self-contained.
    ///
    /// Three constraints, applied in order:
    ///
    /// 1. **Token cap (soft)** — push split forward to keep the
    ///    retained tail under `max_tokens` of estimated content. The
    ///    retained-tail cap is a soft hint; a tool chain that doesn't
    ///    fit gets retained anyway because chain safety is hard.
    /// 2. **Pair safety (hard)** — shift split backward to keep
    ///    `assistant_with_tool_use` and the immediately following
    ///    `user_with_tool_result` together. Catches the common case
    ///    where the boundary lands inside a single tool turn.
    /// 3. **Chain safety (hard)** — advance split forward past any
    ///    leading `user_with_tool_result` whose `tool_use_id` isn't
    ///    in the rest of `to_keep`. Catches the case pair-preservation
    ///    can't see: when the message immediately before the original
    ///    boundary is text-only (e.g. a `summary_ack` from a prior
    ///    compaction), pair-preservation has nothing to anchor on
    ///    and silently leaves the orphan in `to_keep[0]`. The wire
    ///    payload would then start `[summary, summary_ack,
    ///    user(orphan_tool_result), …]` — which Anthropic rejects
    ///    with `messages.2.content.0: unexpected tool_use_id`. Step
    ///    3 makes the split-point selection responsible for chain
    ///    integrity instead of post-hoc stripping the output.
    ///
    /// Step 2 and step 3 can pull in opposite directions (step 2
    /// shifts back, step 3 shifts forward), so the function applies
    /// step 3 last: pair-safety puts the candidate as far back as
    /// it needs to go, then chain-safety advances past any leading
    /// orphan that survived because the immediate prev was text-only.
    fn split_point_preserves_tool_pairs_with_cap(
        messages: &[Message],
        split_point: usize,
        max_tokens: usize,
    ) -> usize {
        let cap_limit = Self::retain_tail_with_token_cap(messages, split_point, max_tokens);
        let pair_safe = Self::split_point_preserves_tool_pairs(messages, cap_limit);
        Self::split_point_skips_leading_orphan(messages, pair_safe)
    }

    /// Advance `split_point` forward until `to_keep[0]` doesn't
    /// contain an orphan `tool_result` block — i.e. a `tool_result`
    /// whose `tool_use_id` isn't satisfied by some `tool_use` block
    /// in `to_keep`.
    ///
    /// Implements step 3 of `split_point_preserves_tool_pairs_with_cap`
    /// (chain safety). Pair-preservation alone can't catch the
    /// "synthetic `summary_ack` precedes an orphan" shape because it
    /// only inspects the immediate prev/next pair; this helper
    /// inspects whether `to_keep[0]`'s `tool_result` blocks point
    /// anywhere `to_keep` will host a matching `tool_use`. When they
    /// don't, the `tool_result` belongs in `to_summarize` (where it
    /// gets text-ified into the summary prose), not in `to_keep`.
    ///
    /// Walks at most `messages.len()` steps because each iteration
    /// advances `split_point` by at least 1.
    fn split_point_skips_leading_orphan(messages: &[Message], mut split_point: usize) -> usize {
        while split_point < messages.len() {
            if Self::leading_message_has_orphan_tool_result(&messages[split_point..]) {
                split_point = split_point.saturating_add(1);
                continue;
            }
            break;
        }
        split_point
    }

    /// True when `to_keep[0]` is a `user` message whose `tool_result`
    /// blocks reference at least one `tool_use_id` not present in
    /// `to_keep`. The check is scoped to the first message because
    /// well-formed Anthropic conversations always have `tool_use`
    /// immediately before `tool_result` — an orphan deeper than
    /// `to_keep[0]` would require the input itself to be malformed
    /// upstream of compaction, which is out of scope here.
    fn leading_message_has_orphan_tool_result(to_keep: &[Message]) -> bool {
        let Some(first) = to_keep.first() else {
            return false;
        };
        let Content::Blocks(blocks) = &first.content else {
            return false;
        };

        // Pull the tool_result ids that appear in the first message.
        // If there are none, the first message can't contribute an
        // orphan and we're done early without scanning the tail.
        let mut needed: Vec<&str> = Vec::new();
        for block in blocks {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                needed.push(tool_use_id.as_str());
            }
        }
        if needed.is_empty() {
            return false;
        }

        // Build the set of tool_use ids `to_keep` will host.
        let known_ids: std::collections::HashSet<&str> = to_keep
            .iter()
            .flat_map(|message| match &message.content {
                Content::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                Content::Text(_) => Vec::new(),
            })
            .collect();

        needed.iter().any(|id| !known_ids.contains(id))
    }

    /// Keep most recent messages that fit within the retained-message token budget.
    fn retain_tail_with_token_cap(messages: &[Message], start: usize, max_tokens: usize) -> usize {
        if start >= messages.len() {
            return messages.len();
        }

        if max_tokens == 0 {
            return messages.len();
        }

        let mut used = 0usize;
        let mut retained_start = messages.len();

        for idx in (start..messages.len()).rev() {
            let message_tokens = TokenEstimator::estimate_message(&messages[idx]);
            if used + message_tokens > max_tokens {
                break;
            }

            retained_start = idx;
            used += message_tokens;
        }

        retained_start
    }

    /// Format messages for summarization.
    ///
    /// Borrows each message rather than taking a slice of owned values so the
    /// caller can pass a filtered view (`Vec<&Message>`) without cloning.
    fn format_messages_for_summary<'a>(messages: impl IntoIterator<Item = &'a Message>) -> String {
        let mut output = String::new();

        for message in messages {
            let role = match message.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
            };

            let _ = write!(output, "{role}: ");

            match &message.content {
                Content::Text(text) => {
                    let _ = writeln!(output, "{text}");
                }
                Content::Blocks(blocks) => {
                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => {
                                let _ = writeln!(output, "{text}");
                            }
                            ContentBlock::Thinking { thinking, .. } => {
                                // Include thinking in summaries for context
                                let _ = writeln!(output, "[Thinking: {thinking}]");
                            }
                            ContentBlock::RedactedThinking { .. } => {
                                let _ = writeln!(output, "[Redacted thinking]");
                            }
                            ContentBlock::OpaqueReasoning { .. } => {
                                // Provider state is deliberately not rendered
                                // into a summarization prompt. Moving or
                                // paraphrasing it would both expose the opaque
                                // payload and break exact replay semantics.
                                let _ = writeln!(output, "[Opaque reasoning state omitted]");
                            }
                            ContentBlock::ToolUse { name, input, .. } => {
                                let _ = writeln!(
                                    output,
                                    "[Called tool: {name} with input: {}]",
                                    serde_json::to_string(input).unwrap_or_default()
                                );
                            }
                            ContentBlock::ToolResult {
                                content, is_error, ..
                            } => {
                                let status = if is_error.unwrap_or(false) {
                                    "error"
                                } else {
                                    "success"
                                };
                                // Truncate long tool results (Unicode-safe; avoid slicing mid-codepoint)
                                let truncated = if content.chars().count() > MAX_TOOL_RESULT_CHARS {
                                    let prefix: String =
                                        content.chars().take(MAX_TOOL_RESULT_CHARS).collect();
                                    format!("{prefix}... (truncated)")
                                } else {
                                    content.clone()
                                };
                                let _ = writeln!(output, "[Tool result ({status}): {truncated}]");
                            }
                            ContentBlock::Image { source } => {
                                let _ = writeln!(output, "[Image: {}]", source.media_type);
                            }
                            ContentBlock::Document { source } => {
                                let _ = writeln!(output, "[Document: {}]", source.media_type);
                            }
                            // `ContentBlock` is `#[non_exhaustive]`; render an
                            // unknown future block kind with a generic marker.
                            _ => {
                                let _ = writeln!(output, "[Unrecognized content block]");
                            }
                        }
                    }
                }
            }
            output.push('\n');
        }

        output
    }

    /// Build the summarization prompt.
    ///
    /// When `prior_summaries` is non-empty (a re-compaction is folding earlier
    /// summaries back in), their prose is prepended as a labeled section so the
    /// model preserves and subsumes those facts in the new summary rather than
    /// losing all pre-first-compaction context.
    fn build_summary_prompt(&self, prior_summaries: &[String], messages_text: &str) -> String {
        let base = format!(
            "{}{}{}",
            self.summary_prompt_prefix, messages_text, self.summary_prompt_suffix
        );

        if prior_summaries.is_empty() {
            return base;
        }

        let prior = prior_summaries.join("\n\n");
        format!(
            "Previous summary of earlier conversation. Preserve every fact below \
             in your new summary so no earlier context is lost:\n{prior}\n\n{base}"
        )
    }

    /// Run a single summarization LLM call, applying the configured
    /// guardrail hooks around it.
    ///
    /// The returned [`SummarizationCall`] reports whether the response hit
    /// the `max_tokens` budget and carries the provider-billed usage so the
    /// caller can surface compaction spend to the agent loop's budgets.
    async fn run_summarization(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<SummarizationCall, SummarizationFailure> {
        let mut request = ChatRequest {
            system: self.system_prompt.clone(),
            messages: vec![Message::user(prompt)],
            tools: None,
            max_tokens: u32::try_from(max_tokens).unwrap_or(u32::MAX),
            max_tokens_explicit: true,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
        };

        // Input guardrail: the summarization call goes through the same
        // `pre_llm_request` hook as regular turns, so compaction cannot be
        // used to smuggle history past a request policy.
        if let Some(hooks) = &self.hooks {
            match hooks.pre_llm_request(&request).await {
                RequestDecision::Modify(modified) => request = *modified,
                RequestDecision::Block(reason) => {
                    return Err(SummarizationFailure {
                        error: anyhow::anyhow!(
                            "Summarization request blocked by guardrail: {reason}"
                        ),
                        usage: TokenUsage::default(),
                    });
                }
                // `Proceed`, plus any future `#[non_exhaustive]` variant,
                // sends the request unchanged (mirrors the agent loop).
                _ => {}
            }
        }

        let outcome = self
            .provider
            .chat(request)
            .await
            .context("Failed to call LLM for summarization")
            .map_err(|error| SummarizationFailure {
                error,
                usage: TokenUsage::default(),
            })?;

        match outcome {
            ChatOutcome::Success(response) => {
                // The provider billed this call regardless of what happens
                // next, so its usage rides every outcome — including
                // guardrail rejections — for budget accounting.
                let usage = TokenUsage {
                    input_tokens: response.usage.input_tokens,
                    output_tokens: response.usage.output_tokens,
                    cached_input_tokens: response.usage.cached_input_tokens,
                    cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
                };
                // Output guardrail: a rejected summary aborts the compaction
                // attempt so it is never persisted. `RetryWithFeedback` is
                // deliberately NOT retried here — the compactor has no
                // feedback loop, and honoring it would let a
                // deterministically-rejecting hook start a paid retry loop
                // inside compaction.
                if let Some(hooks) = &self.hooks {
                    match hooks.on_llm_response(&response).await {
                        ResponseDecision::Block(reason) => {
                            return Err(SummarizationFailure {
                                error: anyhow::anyhow!(
                                    "Summarization response blocked by guardrail: {reason}"
                                ),
                                usage,
                            });
                        }
                        ResponseDecision::RetryWithFeedback(reason) => {
                            return Err(SummarizationFailure {
                                error: anyhow::anyhow!(
                                    "Summarization response rejected by guardrail \
                                     (RetryWithFeedback is not retried during compaction): {reason}"
                                ),
                                usage,
                            });
                        }
                        // `Accept`, plus any future `#[non_exhaustive]`
                        // variant, keeps the summary (mirrors the agent loop).
                        _ => {}
                    }
                }
                let truncated = response.stop_reason == Some(StopReason::MaxTokens);
                let Some(text) = response.first_text().map(String::from) else {
                    return Err(SummarizationFailure {
                        error: anyhow::anyhow!("No text in summarization response"),
                        usage,
                    });
                };
                Ok(SummarizationCall {
                    text,
                    truncated,
                    usage,
                })
            }
            ChatOutcome::RateLimited(_) => Err(SummarizationFailure {
                error: anyhow::anyhow!("Rate limited during summarization"),
                usage: TokenUsage::default(),
            }),
            ChatOutcome::InvalidRequest(msg) => Err(SummarizationFailure {
                error: anyhow::anyhow!("Invalid request during summarization: {msg}"),
                usage: TokenUsage::default(),
            }),
            ChatOutcome::ServerError(msg) => Err(SummarizationFailure {
                error: anyhow::anyhow!("Server error during summarization: {msg}"),
                usage: TokenUsage::default(),
            }),
            // `ChatOutcome` is `#[non_exhaustive]`; an unrecognized outcome
            // fails the summarization rather than returning an empty summary.
            _ => Err(SummarizationFailure {
                error: anyhow::anyhow!("Unrecognized provider outcome during summarization"),
                usage: TokenUsage::default(),
            }),
        }
    }

    /// Summarize `messages`, tracking the provider-billed usage of every
    /// LLM call made (including the enlarged-budget retry on truncation).
    ///
    /// This is the usage-aware core behind both the
    /// [`ContextCompactor::compact`] trait method (which discards the usage
    /// for backward compatibility) and [`ContextCompactor::compact_history`]
    /// (which surfaces it via [`CompactionResult::llm_usage`]).
    async fn summarize_with_usage(
        &self,
        messages: &[Message],
    ) -> Result<(String, TokenUsage), SummarizationFailure> {
        // Separate prior compaction summaries (whose prose must be carried
        // forward) from fresh messages (which still need summarizing). Prior
        // summaries used to be filtered out and silently dropped, destroying
        // all context from before the previous compaction.
        let mut prior_summaries: Vec<String> = Vec::new();
        let mut fresh: Vec<&Message> = Vec::new();
        for message in messages {
            if let Some(text) = Self::extract_summary_text(&message.content) {
                if !text.is_empty() {
                    prior_summaries.push(text);
                }
            } else {
                fresh.push(message);
            }
        }

        // Nothing fresh to summarize: carry prior summaries forward verbatim
        // (no LLM call needed) rather than discarding them.
        if fresh.is_empty() {
            if prior_summaries.is_empty() {
                return Ok((COMPACT_EMPTY_SUMMARY.to_string(), TokenUsage::default()));
            }
            return Ok((prior_summaries.join("\n\n"), TokenUsage::default()));
        }

        let messages_text = Self::format_messages_for_summary(fresh.iter().copied());
        let prompt = self.build_summary_prompt(&prior_summaries, &messages_text);

        let budget = self.config.summary_max_tokens;
        let first = self.run_summarization(prompt.clone(), budget).await?;
        let mut summary = first.text;
        let mut total_usage = first.usage;

        if first.truncated {
            log::warn!(
                "compaction summary hit the max_tokens budget ({budget}); \
                 retrying with a larger budget to avoid silent context loss"
            );
            let retry = match self
                .run_summarization(prompt, budget.saturating_mul(2))
                .await
            {
                Ok(retry) => retry,
                Err(mut failure) => {
                    // The first (truncated) call was still billed: carry its
                    // usage on the failure so the caller can account it.
                    failure.usage.add(&total_usage);
                    return Err(failure);
                }
            };
            total_usage.add(&retry.usage);
            summary = retry.text;
            if retry.truncated {
                log::warn!(
                    "compaction summary still truncated after retry; appending a \
                     truncation marker so downstream context loss is visible"
                );
                summary.push_str(TRUNCATED_SUMMARY_MARKER);
            }
        }

        Ok((summary, total_usage))
    }
}

/// Outcome of one summarization LLM round-trip.
struct SummarizationCall {
    text: String,
    truncated: bool,
    usage: TokenUsage,
}

/// A failed summarization round-trip, carrying whatever usage was billed
/// before the failure (a response rejected by the output guardrail was
/// still billed; a request blocked before dispatch was not).
struct SummarizationFailure {
    error: anyhow::Error,
    usage: TokenUsage,
}

impl<P: LlmProvider + ?Sized, H: AgentHooks> LlmContextCompactor<P, H> {
    /// Usage-aware core of [`ContextCompactor::compact_history`]: a failure
    /// carries the billed usage of any summarization calls already made.
    async fn compact_history_inner(
        &self,
        mut messages: Vec<Message>,
    ) -> Result<CompactionResult, FailedCompaction> {
        let original_count = messages.len();
        let original_tokens = self.estimate_tokens(&messages);

        // OpenAI Responses reasoning items (and any future provider-owned
        // opaque state) must remain in their original history position for a
        // valid follow-up. Generic compaction turns older messages into prose,
        // so it cannot preserve that contract safely.
        if Self::has_opaque_reasoning(&messages) {
            log::debug!(
                "skipping generic context compaction for history with opaque provider reasoning state"
            );
            return Ok(CompactionResult {
                messages,
                original_count,
                new_count: original_count,
                original_tokens,
                new_tokens: original_tokens,
                llm_usage: TokenUsage::default(),
            });
        }

        // Ensure we have enough messages to compact
        if messages.len() <= self.config.retain_recent {
            return Ok(CompactionResult {
                messages,
                original_count,
                new_count: original_count,
                original_tokens,
                new_tokens: original_tokens,
                llm_usage: TokenUsage::default(),
            });
        }

        // Split messages: old messages to summarize, recent messages to keep
        let mut split_point = messages.len().saturating_sub(self.config.retain_recent);
        split_point = Self::split_point_preserves_tool_pairs_with_cap(
            &messages,
            split_point,
            self.config.max_retained_tail_tokens,
        );

        // Move the retained tail out of `messages` so it doesn't have to be
        // cloned: `messages` then holds exactly the slice to summarize.
        let to_keep = messages.split_off(split_point);
        let to_summarize = messages;

        // Summarize old messages
        let (summary, llm_usage) =
            self.summarize_with_usage(&to_summarize)
                .await
                .map_err(|failure| FailedCompaction {
                    error: failure.error,
                    llm_usage: failure.usage,
                })?;

        // Build new message history
        let mut new_messages = Vec::with_capacity(2 + to_keep.len());

        // Add summary as a user message
        new_messages.push(Message::user(format!("{SUMMARY_PREFIX}{summary}")));

        // Add acknowledgment from assistant only when some recent tail remains.
        // If compaction drops the entire retained tail due to the token cap, ending
        // the request with this synthetic assistant message would act like assistant
        // prefill and Anthropic rejects that shape.
        if !to_keep.is_empty() {
            new_messages.push(Message::assistant(SUMMARY_ACKNOWLEDGMENT));
        }

        // Add recent messages. `to_keep` is guaranteed self-consistent
        // by `split_point_preserves_tool_pairs_with_cap` (steps 2 and
        // 3): any orphan `tool_result` was either folded into the
        // summary (split shifted forward) or paired with its
        // `tool_use` inside `to_keep` (split shifted backward). No
        // post-hoc rewriting of the assembled output is required.
        // The tail is moved (not cloned) since `compact_history` owns it.
        new_messages.extend(to_keep);

        let new_count = new_messages.len();
        let new_tokens = self.estimate_tokens(&new_messages);

        Ok(CompactionResult {
            messages: new_messages,
            original_count,
            new_count,
            original_tokens,
            new_tokens,
            llm_usage,
        })
    }
}

#[async_trait]
impl<P: LlmProvider + ?Sized, H: AgentHooks> ContextCompactor for LlmContextCompactor<P, H> {
    async fn compact(&self, messages: &[Message]) -> Result<String> {
        let (summary, _usage) = self
            .summarize_with_usage(messages)
            .await
            .map_err(|failure| failure.error)?;
        Ok(summary)
    }

    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        TokenEstimator::estimate_history(messages)
    }

    fn needs_compaction(&self, messages: &[Message]) -> bool {
        if !self.config.auto_compact {
            return false;
        }

        // Provider-owned opaque reasoning state has exact replay semantics.
        // Do not repeatedly schedule a generic compaction that must refuse to
        // rewrite this history; a provider-native compactor can replace this
        // guard when it becomes available.
        if Self::has_opaque_reasoning(messages) {
            return false;
        }

        if messages.len() < self.config.min_messages_for_compaction {
            return false;
        }

        let estimated_tokens = self.estimate_tokens(messages);
        estimated_tokens > self.config.threshold_tokens
    }

    async fn compact_history(&self, messages: Vec<Message>) -> Result<CompactionResult> {
        self.compact_history_inner(messages)
            .await
            .map_err(|failure| failure.error)
    }

    async fn compact_history_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> Result<CompactionResult, FailedCompaction> {
        self.compact_history_inner(messages).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatResponse, StopReason, Usage};
    use anyhow::bail;
    use std::sync::Mutex;

    struct MockProvider {
        summary_response: String,
        requests: Arc<Mutex<Vec<String>>>,
        /// When true, the summary echoes the received prompt (simulating an LLM
        /// that faithfully preserves its input — used to assert carry-forward).
        echo_input: bool,
        /// `stop_reason` returned by the mock; `MaxTokens` simulates truncation.
        stop_reason: StopReason,
    }

    impl MockProvider {
        fn build(
            summary: &str,
            requests: Arc<Mutex<Vec<String>>>,
            echo_input: bool,
            stop_reason: StopReason,
        ) -> Self {
            Self {
                summary_response: summary.to_string(),
                requests,
                echo_input,
                stop_reason,
            }
        }

        fn new(summary: &str) -> Self {
            Self::build(
                summary,
                Arc::new(Mutex::new(Vec::new())),
                false,
                StopReason::EndTurn,
            )
        }

        fn new_with_request_log(summary: &str, requests: Arc<Mutex<Vec<String>>>) -> Self {
            Self::build(summary, requests, false, StopReason::EndTurn)
        }

        /// A provider whose summary echoes the received prompt verbatim.
        fn new_echo(requests: Arc<Mutex<Vec<String>>>) -> Self {
            Self::build("", requests, true, StopReason::EndTurn)
        }

        /// A provider that always reports `MaxTokens` (a truncated summary).
        fn new_truncating(summary: &str, requests: Arc<Mutex<Vec<String>>>) -> Self {
            Self::build(summary, requests, false, StopReason::MaxTokens)
        }

        fn user_prompt_of(request: &ChatRequest) -> String {
            request
                .messages
                .iter()
                .find_map(|message| match &message.content {
                    Content::Text(text) => Some(text.clone()),
                    Content::Blocks(blocks) => {
                        let text = blocks
                            .iter()
                            .filter_map(|block| {
                                if let ContentBlock::Text { text } = block {
                                    Some(text.as_str())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if text.is_empty() { None } else { Some(text) }
                    }
                })
                .unwrap_or_default()
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
            let user_prompt = Self::user_prompt_of(&request);
            if let Ok(mut entries) = self.requests.lock() {
                entries.push(user_prompt.clone());
            }
            let text = if self.echo_input {
                user_prompt
            } else {
                self.summary_response.clone()
            };
            Ok(ChatOutcome::Success(ChatResponse {
                id: "test".to_string(),
                content: vec![ContentBlock::Text { text }],
                model: "mock".to_string(),
                stop_reason: Some(self.stop_reason),
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            }))
        }

        fn model(&self) -> &'static str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    #[test]
    fn test_needs_compaction_below_threshold() {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default()
            .with_threshold_tokens(10_000)
            .with_min_messages(5);
        let compactor = LlmContextCompactor::new(provider, config);

        // Only 3 messages, below min_messages
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi"),
            Message::user("How are you?"),
        ];

        assert!(!compactor.needs_compaction(&messages));
    }

    #[test]
    fn test_needs_compaction_above_threshold() {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default()
            .with_threshold_tokens(50) // Very low threshold
            .with_min_messages(3);
        let compactor = LlmContextCompactor::new(provider, config);

        // Messages that exceed threshold
        let messages = vec![
            Message::user("Hello, this is a longer message to test compaction"),
            Message::assistant(
                "Hi there! This is also a longer response to help trigger compaction",
            ),
            Message::user("Great, let's continue with even more text here"),
            Message::assistant("Absolutely, adding more content to ensure we exceed the threshold"),
        ];

        assert!(compactor.needs_compaction(&messages));
    }

    #[test]
    fn test_needs_compaction_auto_disabled() {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default()
            .with_threshold_tokens(10) // Very low
            .with_min_messages(1)
            .with_auto_compact(false);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user("Hello, this is a longer message"),
            Message::assistant("Response here"),
        ];

        assert!(!compactor.needs_compaction(&messages));
    }

    #[test]
    fn summary_prompt_redacts_opaque_reasoning_payload() {
        let secret = "opaque-secret-that-must-not-enter-the-summary-prompt";
        let message = Message::assistant_with_content(vec![ContentBlock::OpaqueReasoning {
            provider: "test-provider".to_owned(),
            data: serde_json::json!({"encrypted_content": secret}),
        }]);

        let rendered = LlmContextCompactor::<MockProvider>::format_messages_for_summary([&message]);
        assert!(rendered.contains("[Opaque reasoning state omitted]"));
        assert!(!rendered.contains(secret));
    }

    #[tokio::test]
    async fn compact_history_preserves_opaque_reasoning_in_retained_tail() -> Result<()> {
        let provider = Arc::new(MockProvider::new("older context"));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(3);
        let compactor = LlmContextCompactor::new(provider, config);
        let opaque_data = serde_json::json!({
            "id": "reasoning_1",
            "encrypted_content": "ciphertext"
        });
        let messages = vec![
            Message::user("old question"),
            Message::assistant("old answer"),
            Message::user("current question"),
            Message::assistant_with_content(vec![ContentBlock::OpaqueReasoning {
                provider: "test-provider".to_owned(),
                data: opaque_data.clone(),
            }]),
        ];

        let result = compactor.compact_history(messages).await?;
        let retained = result
            .messages
            .last()
            .context("compacted history should retain the newest assistant message")?;
        let Content::Blocks(blocks) = &retained.content else {
            bail!("retained assistant message should contain blocks");
        };
        assert!(matches!(
            blocks.first(),
            Some(ContentBlock::OpaqueReasoning { provider, data })
                if provider == "test-provider" && data == &opaque_data
        ));
        Ok(())
    }

    #[tokio::test]
    async fn compact_history_bypasses_opaque_reasoning_before_the_split() -> Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "summary that must not be requested",
            Arc::clone(&requests),
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(1)
            .with_min_messages(1)
            .with_threshold_tokens(1);
        let compactor = LlmContextCompactor::new(provider, config);
        let opaque_data = serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "ciphertext"
        });
        let messages = vec![
            Message::user("older user context"),
            // This message lies before the normal `len - retain_recent` split
            // and would previously have been converted into a prose summary.
            Message::assistant_with_content(vec![
                ContentBlock::OpaqueReasoning {
                    provider: "openai-responses".to_owned(),
                    data: opaque_data,
                },
                ContentBlock::Text {
                    text: "older assistant response".to_owned(),
                },
            ]),
            Message::user("newer user context"),
            Message::assistant("newer assistant response"),
        ];
        let serialized_before = serde_json::to_value(&messages)?;

        assert!(
            !compactor.needs_compaction(&messages),
            "generic compaction must not be scheduled for opaque replay state"
        );

        let result = compactor.compact_history(messages).await?;

        assert_eq!(serde_json::to_value(&result.messages)?, serialized_before);
        assert_eq!(result.new_count, result.original_count);
        assert_eq!(result.new_tokens, result.original_tokens);
        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert!(
            recorded.is_empty(),
            "opaque replay state must not be sent to the generic summarizer"
        );
        drop(recorded);
        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history() -> Result<()> {
        let provider = Arc::new(MockProvider::new(
            "User asked about Rust programming. Assistant explained ownership, borrowing, and lifetimes.",
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(3);
        let compactor = LlmContextCompactor::new(provider, config);

        // Use longer messages to ensure compaction actually reduces tokens
        let messages = vec![
            Message::user(
                "What is Rust? I've heard it's a systems programming language but I don't know much about it. Can you explain the key features and why people are excited about it?",
            ),
            Message::assistant(
                "Rust is a systems programming language focused on safety, speed, and concurrency. It achieves memory safety without garbage collection through its ownership system. The key features include zero-cost abstractions, guaranteed memory safety, threads without data races, and minimal runtime.",
            ),
            Message::user(
                "Tell me about ownership in detail. How does it work and what are the rules? I want to understand this core concept thoroughly.",
            ),
            Message::assistant(
                "Ownership is Rust's central feature with three rules: each value has one owner, only one owner at a time, and the value is dropped when owner goes out of scope. This system prevents memory leaks, double frees, and dangling pointers at compile time.",
            ),
            Message::user("What about borrowing?"), // Keep
            Message::assistant("Borrowing allows references to data without taking ownership."), // Keep
        ];

        let result = compactor.compact_history(messages).await?;

        // Should have: summary message + ack + 2 recent messages = 4
        assert_eq!(result.new_count, 4);
        assert_eq!(result.original_count, 6);

        // With longer original messages, compaction should reduce tokens
        assert!(
            result.new_tokens < result.original_tokens,
            "Expected fewer tokens after compaction: new={} < original={}",
            result.new_tokens,
            result.original_tokens
        );

        // First message should be the summary
        if let Content::Text(text) = &result.messages[0].content {
            assert!(text.contains("Previous conversation summary"));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_too_few_messages() -> Result<()> {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default().with_retain_recent(5);
        let compactor = LlmContextCompactor::new(provider, config);

        // Only 3 messages, less than retain_recent
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi"),
            Message::user("Bye"),
        ];

        let result = compactor.compact_history(messages.clone()).await?;

        // Should return original messages unchanged
        assert_eq!(result.new_count, 3);
        assert_eq!(result.messages.len(), 3);

        Ok(())
    }

    #[test]
    fn test_format_messages_for_summary() {
        let messages = vec![Message::user("Hello"), Message::assistant("Hi there!")];

        let formatted = LlmContextCompactor::<MockProvider>::format_messages_for_summary(&messages);

        assert!(formatted.contains("User: Hello"));
        assert!(formatted.contains("Assistant: Hi there!"));
    }

    #[test]
    fn test_format_messages_for_summary_truncates_tool_results_unicode_safely() {
        let long_unicode = "é".repeat(600);

        let messages = vec![Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "tool-1".to_string(),
                content: long_unicode,
                is_error: Some(false),
            }]),
        }];

        let formatted = LlmContextCompactor::<MockProvider>::format_messages_for_summary(&messages);

        assert!(formatted.contains("... (truncated)"));
    }

    #[tokio::test]
    async fn test_compact_carries_prior_summary_into_request() -> Result<()> {
        // A prior compaction summary must be carried forward into the
        // summarization input (not silently filtered out), so its facts are
        // preserved across re-compaction. The fresh message is summarized as
        // usual; the prior summary is included as a "Previous summary" section.
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "Fresh summary",
            requests.clone(),
        ));
        let config = CompactionConfig::default().with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user(format!("{SUMMARY_PREFIX}already compacted context")),
            Message::assistant("Continue with the next task using this context."),
        ];

        let summary = compactor.compact(&messages).await?;

        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert_eq!(recorded.len(), 1);
        // The new summary is the LLM's output; the prior summary lives in the
        // request, where a real model subsumes it into the new summary.
        assert_eq!(summary, "Fresh summary");
        assert!(recorded[0].contains("Continue with the next task using this context."));
        assert!(
            recorded[0].contains("already compacted context"),
            "prior summary must be carried into the summarization input"
        );
        drop(recorded);

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_carries_prior_summary_in_candidate_payload() -> Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "Fresh history summary",
            requests.clone(),
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user(format!("{SUMMARY_PREFIX}already compacted context")),
            Message::assistant("Current turn content from the latest exchange."),
            Message::assistant("Recent message that should stay."),
            Message::user("Newest note that should stay."),
        ];

        let result = compactor.compact_history(messages).await?;

        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert_eq!(recorded.len(), 1);
        assert!(recorded[0].contains("Current turn content from the latest exchange."));
        // The prior summary is carried into the summarization input rather than
        // being silently discarded.
        assert!(
            recorded[0].contains("already compacted context"),
            "prior summary content must reach the summarizer"
        );
        drop(recorded);
        assert_eq!(result.new_count, 4);

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_carries_summaries_forward_when_window_has_only_summaries()
    -> Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "This summary should not be used",
            requests.clone(),
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user(format!("{SUMMARY_PREFIX}first prior compacted section")),
            Message::assistant(format!("{SUMMARY_PREFIX}second prior compacted section")),
            Message::user(format!("{SUMMARY_PREFIX}third prior compacted section")),
            Message::assistant("final short note"),
        ];

        let result = compactor.compact_history(messages).await?;

        // No fresh content in the candidate window -> no LLM call is made, but
        // the prior summaries must be carried forward verbatim, NOT replaced
        // with an empty-summary placeholder (which used to destroy context).
        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert!(recorded.is_empty());
        drop(recorded);
        assert_eq!(result.new_count, 4);
        assert_eq!(result.messages.len(), 4);

        if let Content::Text(text) = &result.messages[0].content {
            assert!(
                text.contains("first prior compacted section"),
                "first prior summary lost"
            );
            assert!(
                text.contains("second prior compacted section"),
                "second prior summary lost"
            );
            assert!(!text.contains(COMPACT_EMPTY_SUMMARY));
        } else {
            panic!("Expected summary text in first message");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_preserves_tool_use_tool_result_pairs() -> Result<()> {
        let provider = Arc::new(MockProvider::new("Summary of earlier conversation."));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(3);
        let compactor = LlmContextCompactor::new(provider, config);

        // Build a history where the split_point (len - retain_recent = 5 - 2 = 3)
        // would land exactly on the user tool_result message at index 3,
        // which would orphan it from its assistant tool_use at index 2.
        let messages = vec![
            // index 0: user
            Message::user("What files are in the project?"),
            // index 1: assistant text
            Message::assistant("Let me check that for you."),
            // index 2: assistant with tool_use
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "list_files".to_string(),
                    input: serde_json::json!({}),
                    thought_signature: None,
                }]),
            },
            // index 3: user with tool_result (naive split would land here)
            Message {
                role: Role::User,
                content: Content::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "tool_1".to_string(),
                    content: "file1.rs\nfile2.rs".to_string(),
                    is_error: None,
                }]),
            },
            // index 4: assistant final response
            Message::assistant("The project contains file1.rs and file2.rs."),
        ];

        let result = compactor.compact_history(messages).await?;

        // The split_point should have been adjusted back from 3 to 2,
        // so to_keep includes: [assistant tool_use, user tool_result, assistant response]
        // Plus summary + ack = 5 total
        assert_eq!(result.new_count, 5);

        // Verify the kept messages include the tool_use/tool_result pair
        // After summary + ack, the third message should be the assistant with tool_use
        let kept_assistant = &result.messages[2];
        if let Content::Blocks(blocks) = &kept_assistant.content {
            assert!(
                blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolUse { .. })),
                "Expected assistant tool_use in kept messages"
            );
        } else {
            panic!("Expected Blocks content for assistant tool_use message");
        }

        // The fourth message should be the user tool_result
        let kept_user = &result.messages[3];
        if let Content::Blocks(blocks) = &kept_user.content {
            assert!(
                blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. })),
                "Expected user tool_result in kept messages"
            );
        } else {
            panic!("Expected Blocks content for user tool_result message");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_split_skips_leading_orphan_after_summary_ack() -> Result<()> {
        // The user-visible bug at M7.5: a previously
        // compacted history was re-compacted in a later turn. The
        // first compaction left
        // `[summary, summary_ack, user(tool_result toolu_X),
        //  assistant(toolu_X reply), ...]`. On the second pass the
        // default `split_point` (len - retain_recent = 5 - 3 = 2)
        // would have made `to_keep[0] == user(tool_result toolu_X)`,
        // and the synthetic `[summary, summary_ack, …]` prefix the
        // compactor inserts in front of `to_keep` has no `tool_use`
        // blocks — so the next request to Anthropic blew up with
        // `messages.2.content.0: unexpected tool_use_id`.
        //
        // Pair-preservation alone can't fix this: it only inspects
        // the immediate prev/next pair (here `summary_ack` vs
        // `user(tool_result)`) and `summary_ack` is text-only, so the
        // pair check sees no `tool_use` to anchor on and lets the
        // orphan through. The chain-safety pass added in
        // `split_point_preserves_tool_pairs_with_cap` step 3 walks
        // the candidate forward past any leading orphan, so the
        // `tool_result` lands in `to_summarize` and gets folded into
        // the summary's prose where it's harmless.
        //
        // The assertion is structural, not block-counting: every
        // surviving `tool_result` must reference a `tool_use` that
        // appears earlier in the new message list. No
        // post-compaction stripping is involved — the split point
        // alone is responsible for chain integrity.
        let provider = Arc::new(MockProvider::new("Re-summary."));
        let config = CompactionConfig::default()
            .with_retain_recent(3)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user(format!("{SUMMARY_PREFIX}Old summary about toolu_X.")),
            Message::assistant(SUMMARY_ACKNOWLEDGMENT),
            Message {
                role: Role::User,
                content: Content::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_X".to_string(),
                    content: "result for X".to_string(),
                    is_error: None,
                }]),
            },
            Message::assistant("Result interpreted."),
            Message::user("Now what?"),
        ];

        let result = compactor.compact_history(messages).await?;

        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for msg in &result.messages {
            if let Content::Blocks(blocks) = &msg.content {
                for block in blocks {
                    match block {
                        ContentBlock::ToolResult { tool_use_id, .. } => {
                            assert!(
                                seen_ids.contains(tool_use_id),
                                "orphan tool_use_id {tool_use_id} survived split selection",
                            );
                        }
                        ContentBlock::ToolUse { id, .. } => {
                            seen_ids.insert(id.clone());
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_keeps_tool_pair_when_immediate_prev_is_text_only() -> Result<()> {
        // Tighter regression for the chain-safety boundary: even
        // when the message *before* the candidate split point is
        // text-only (so pair-preservation has nothing to anchor on),
        // chain-safety must shift the split forward past a leading
        // `user(tool_result)` whose `tool_use` would otherwise be
        // folded into the summary.
        let provider = Arc::new(MockProvider::new("Boundary summary."));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        // Layout (5 messages, retain_recent=2 → initial split=3):
        //   0: user("first turn") — to_summarize
        //   1: assistant("text only") — to_summarize, immediate prev
        //   2: user(tool_result toolu_Y) — orphan in default to_keep
        //   3: assistant("then a reply")
        //   4: user("ok thanks")
        //
        // The corresponding `tool_use` for toolu_Y was lost long
        // ago — there's no `tool_use` anywhere in `messages`. With
        // pair-preservation alone, `to_keep` would start at index 3
        // (or 2 unshifted), leaving the orphan at the head and
        // tripping Anthropic.
        let messages = vec![
            Message::user("first turn"),
            Message::assistant("text only"),
            Message {
                role: Role::User,
                content: Content::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_Y".to_string(),
                    content: "ancient result".to_string(),
                    is_error: None,
                }]),
            },
            Message::assistant("then a reply"),
            Message::user("ok thanks"),
        ];

        let result = compactor.compact_history(messages).await?;

        // No tool_result block survives anywhere — the only one in
        // input was orphaned and the split-shift folded it into the
        // summary.
        let has_tool_result = result.messages.iter().any(|m| {
            matches!(
                &m.content,
                Content::Blocks(blocks)
                    if blocks.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. }))
            )
        });
        assert!(
            !has_tool_result,
            "orphan tool_result should have been pushed into to_summarize, not retained",
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_retained_tail_is_token_capped() -> Result<()> {
        let provider = Arc::new(MockProvider::new(
            "Project summary with a long context and technical context.",
        ));
        let config = CompactionConfig::default()
            .with_retain_recent(8)
            .with_min_messages(1)
            .with_threshold_tokens(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let mut messages = Vec::new();

        // Older messages that will be summarized away.
        messages.extend((0..6).map(|index| Message::user(format!("pre-compaction noise {index}"))));

        // Newer long messages: intentionally large to force retained-tail truncation.
        messages.extend(
            (0..8).map(|index| Message::assistant(format!("kept-{index}: {}", "x".repeat(12_000)))),
        );

        let result = compactor.compact_history(messages).await?;

        // The retained tail should be token capped and therefore shorter than retain_recent.
        let retained_tail = &result.messages[2..];
        assert!(retained_tail.len() < 8);

        let mut latest_index = -1i32;
        let mut all_retained = true;
        for message in retained_tail {
            if let Content::Text(text) = &message.content {
                if let Some(number) = text.split(':').next().and_then(|prefix| {
                    prefix
                        .strip_prefix("kept-")
                        .and_then(|rest| rest.parse::<i32>().ok())
                }) {
                    if number >= 0 {
                        latest_index = latest_index.max(number);
                    }
                } else {
                    all_retained = false;
                }
            } else {
                all_retained = false;
            }
        }

        assert!(all_retained);
        assert_eq!(latest_index, 7);
        assert!(
            TokenEstimator::estimate_history(retained_tail)
                <= compactor.config().max_retained_tail_tokens
        );
        assert!(compactor.needs_compaction(&result.messages));

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_skips_summary_ack_when_retained_tail_is_empty() -> Result<()> {
        let provider = Arc::new(MockProvider::new("Summary for oversized user turn."));
        let config = CompactionConfig::default()
            .with_retain_recent(1)
            .with_min_messages(1)
            .with_threshold_tokens(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::assistant("Earlier assistant context."),
            Message::user(format!("oversized-user-turn: {}", "x".repeat(200_000))),
        ];

        let result = compactor.compact_history(messages).await?;

        assert_eq!(result.new_count, 1);
        assert_eq!(result.messages.len(), 1);

        let only_message = &result.messages[0];
        assert_eq!(only_message.role, Role::User);

        if let Content::Text(text) = &only_message.content {
            assert!(text.contains("Previous conversation summary"));
            assert!(!text.contains(SUMMARY_ACKNOWLEDGMENT));
        } else {
            panic!("Expected summary text when retained tail is empty");
        }

        Ok(())
    }

    fn message_contains(message: &Message, needle: &str) -> bool {
        match &message.content {
            Content::Text(text) => text.contains(needle),
            Content::Blocks(blocks) => blocks.iter().any(|block| match block {
                ContentBlock::Text { text } => text.contains(needle),
                _ => false,
            }),
        }
    }

    #[tokio::test]
    async fn test_epoch_one_facts_survive_two_compactions() -> Result<()> {
        // Regression for re-compaction data loss: a fact recorded before the
        // first compaction must still be present in the history after a second
        // compaction. The echo provider models a faithful LLM that preserves
        // its input; the bug was that the first summary (carrying the epoch-1
        // fact) was filtered out of the second summarization and dropped.
        const EPOCH1_FACT: &str = "EPOCH1_FACT: the API key lives in config/secrets.toml";

        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_echo(requests.clone()));
        let config = CompactionConfig::default()
            .with_retain_recent(2)
            .with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let epoch1 = vec![
            Message::user(EPOCH1_FACT),
            Message::assistant("Understood, noted the secrets path."),
            Message::user("Now add error handling to main.rs."),
            Message::assistant("Added error handling to main.rs."),
            Message::user("latest user message one"),
            Message::assistant("latest assistant message two"),
        ];

        let first = compactor.compact_history(epoch1).await?;
        assert!(
            first
                .messages
                .iter()
                .any(|m| message_contains(m, "EPOCH1_FACT")),
            "epoch-1 fact must be captured in the first summary"
        );

        // Build the epoch-2 history on top of the first compaction's output.
        let mut epoch2 = first.messages;
        epoch2.push(Message::user("Another later turn."));
        epoch2.push(Message::assistant("Reply to the later turn."));
        epoch2.push(Message::user("Final turn a."));
        epoch2.push(Message::assistant("Final turn b."));

        let second = compactor.compact_history(epoch2).await?;

        assert!(
            second
                .messages
                .iter()
                .any(|m| message_contains(m, "EPOCH1_FACT")),
            "epoch-1 fact must survive the second compaction"
        );

        // Sanity: the second compaction actually summarized (made an LLM call
        // on the prior summary), so this is a true re-compaction path.
        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert!(
            recorded.iter().any(|req| req.contains("EPOCH1_FACT")),
            "prior summary carrying the epoch-1 fact must reach the summarizer"
        );
        drop(recorded);

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_history_long_tool_chain_respects_token_cap() -> Result<()> {
        // Regression for the pair-shift bug: in an unbroken tool chain, the
        // old second clause of `crosses_tool_pair` walked the split point back
        // through the entire chain, retaining everything and defeating the
        // token cap. With only the assistant(tool_use)->user(tool_result)
        // boundary unsplittable, the retained tail stays bounded near the cap.
        let provider = Arc::new(MockProvider::new("Summary of the early tool chain."));
        let cap = 20_000;
        // retain_recent asks to keep many messages, but the cap must override
        // it. retain_recent < message count so we don't hit the early return.
        let config = CompactionConfig::default()
            .with_retain_recent(18)
            .with_min_messages(1)
            .with_threshold_tokens(1)
            .with_max_retained_tail_tokens(cap);
        let compactor = LlmContextCompactor::new(provider, config);

        // 10 alternating tool pairs (20 messages), each large enough that the
        // whole chain dwarfs the cap.
        let mut messages = Vec::new();
        for i in 0..10 {
            messages.push(Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![ContentBlock::ToolUse {
                    id: format!("tool_{i}"),
                    name: "run".to_string(),
                    input: serde_json::json!({ "arg": "y".repeat(12_000) }),
                    thought_signature: None,
                }]),
            });
            messages.push(Message {
                role: Role::User,
                content: Content::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: format!("tool_{i}"),
                    content: format!("result-{i}: {}", "z".repeat(12_000)),
                    is_error: None,
                }]),
            });
        }

        let full_tokens = TokenEstimator::estimate_history(&messages);
        assert!(
            full_tokens > cap * 2,
            "test setup: full chain must far exceed the cap"
        );

        let result = compactor.compact_history(messages).await?;

        // The retained tail is non-empty, so the output is
        // [summary, ack, ...tail]; skip the synthetic summary + ack prefix.
        let retained_tail = &result.messages[2..];

        let tail_tokens = TokenEstimator::estimate_history(retained_tail);
        // Bounded near the cap (soft cap allows one extra message from pair
        // preservation); crucially NOT the entire chain.
        assert!(
            tail_tokens <= cap + 8_000,
            "retained tail {tail_tokens} should be bounded by the cap {cap}, not the whole chain"
        );
        assert!(
            retained_tail.len() < 20,
            "compaction must have summarized part of the chain"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_compact_warns_and_marks_truncated_summary() -> Result<()> {
        // Regression for silent summary truncation: when the summarizer hits
        // MaxTokens, the compactor retries with a larger budget and, if still
        // truncated, appends a visible marker instead of silently accepting a
        // clipped summary.
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_truncating(
            "partial summary cut off mid-",
            requests.clone(),
        ));
        let config = CompactionConfig::default().with_min_messages(1);
        let compactor = LlmContextCompactor::new(provider, config);

        let messages = vec![
            Message::user("Some content that needs summarizing."),
            Message::assistant("More content to summarize here."),
        ];

        let summary = compactor.compact(&messages).await?;

        assert!(
            summary.contains("[summary truncated"),
            "a persistently truncated summary must carry a truncation marker"
        );

        // The compactor retried once with a larger budget: two calls total.
        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert_eq!(recorded.len(), 2, "truncation should trigger one retry");
        drop(recorded);

        Ok(())
    }

    /// Long-enough history that `compact_history` performs a real
    /// summarization call (more messages than `retain_recent`).
    fn summarizable_messages() -> Vec<Message> {
        vec![
            Message::user("First question with enough words to summarize meaningfully."),
            Message::assistant("First answer, also carrying plenty of prose to compact."),
            Message::user("Second question continuing the earlier conversation topic."),
            Message::assistant("Second answer expanding on the topic at some length."),
            Message::user("Third question?"),
            Message::assistant("Third answer."),
        ]
    }

    struct BlockRequestHooks;

    #[async_trait]
    impl crate::hooks::AgentHooks for BlockRequestHooks {
        async fn pre_llm_request(&self, _request: &ChatRequest) -> RequestDecision {
            RequestDecision::Block("summaries are not allowed".to_string())
        }
    }

    struct ModifyRequestHooks;

    #[async_trait]
    impl crate::hooks::AgentHooks for ModifyRequestHooks {
        async fn pre_llm_request(&self, request: &ChatRequest) -> RequestDecision {
            let mut modified = request.clone();
            modified.messages = vec![Message::user("MODIFIED_SUMMARY_PROMPT")];
            RequestDecision::Modify(Box::new(modified))
        }
    }

    struct BlockResponseHooks;

    #[async_trait]
    impl crate::hooks::AgentHooks for BlockResponseHooks {
        async fn on_llm_response(&self, _response: &ChatResponse) -> ResponseDecision {
            ResponseDecision::Block("summary leaks a secret".to_string())
        }
    }

    struct RetryResponseHooks;

    #[async_trait]
    impl crate::hooks::AgentHooks for RetryResponseHooks {
        async fn on_llm_response(&self, _response: &ChatResponse) -> ResponseDecision {
            ResponseDecision::RetryWithFeedback("try harder".to_string())
        }
    }

    #[tokio::test]
    async fn blocking_request_hook_aborts_compaction_before_the_llm_call() -> Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "summary",
            requests.clone(),
        ));
        let config = CompactionConfig::default().with_retain_recent(2);
        let compactor = LlmContextCompactor::new(provider, config)
            .with_guardrail_hooks(Arc::new(BlockRequestHooks));

        let error = match compactor.compact_history(summarizable_messages()).await {
            Ok(result) => anyhow::bail!("blocked compaction must not succeed: {result:?}"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("blocked by guardrail"),
            "unexpected error: {error}"
        );

        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert!(
            recorded.is_empty(),
            "a blocked request must never reach the provider"
        );
        drop(recorded);
        Ok(())
    }

    #[tokio::test]
    async fn modify_request_hook_reaches_the_provider() -> Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "summary",
            requests.clone(),
        ));
        let config = CompactionConfig::default().with_retain_recent(2);
        let compactor = LlmContextCompactor::new(provider, config)
            .with_guardrail_hooks(Arc::new(ModifyRequestHooks));

        let result = compactor.compact_history(summarizable_messages()).await?;
        assert!(result.new_count < result.original_count);

        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert_eq!(
            recorded.as_slice(),
            ["MODIFIED_SUMMARY_PROMPT"],
            "the provider must receive the hook-modified request"
        );
        drop(recorded);
        Ok(())
    }

    #[tokio::test]
    async fn blocked_response_aborts_compaction() -> Result<()> {
        let provider = Arc::new(MockProvider::new("a summary that leaks the secret"));
        let config = CompactionConfig::default().with_retain_recent(2);
        let compactor = LlmContextCompactor::new(provider, config)
            .with_guardrail_hooks(Arc::new(BlockResponseHooks));

        let error = match compactor.compact_history(summarizable_messages()).await {
            Ok(result) => anyhow::bail!("blocked summary must not be returned: {result:?}"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("blocked by guardrail"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn retry_with_feedback_response_aborts_compaction_without_retrying() -> Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(MockProvider::new_with_request_log(
            "summary",
            requests.clone(),
        ));
        let config = CompactionConfig::default().with_retain_recent(2);
        let compactor = LlmContextCompactor::new(provider, config)
            .with_guardrail_hooks(Arc::new(RetryResponseHooks));

        let error = match compactor.compact_history(summarizable_messages()).await {
            Ok(result) => anyhow::bail!("rejected summary must not be returned: {result:?}"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("not retried during compaction"),
            "unexpected error: {error}"
        );

        // No paid retry loop: exactly one LLM call was made.
        let recorded = requests
            .lock()
            .map_err(|_| anyhow::anyhow!("request log poisoned"))?;
        assert_eq!(recorded.len(), 1, "RetryWithFeedback must not retry");
        drop(recorded);
        Ok(())
    }

    #[tokio::test]
    async fn compact_history_reports_summarization_usage() -> Result<()> {
        let provider = Arc::new(MockProvider::new("summary"));
        let config = CompactionConfig::default().with_retain_recent(2);
        let compactor = LlmContextCompactor::new(provider, config);

        let result = compactor.compact_history(summarizable_messages()).await?;
        // The mock bills 100 input / 50 output per call; one call was made.
        assert_eq!(result.llm_usage.input_tokens, 100);
        assert_eq!(result.llm_usage.output_tokens, 50);
        Ok(())
    }
}
