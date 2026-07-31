//! Live context-occupancy estimation for in-flight LLM calls (ENG-9510).
//!
//! ## Problem
//!
//! The journal only learns a call's real prompt size when the attempt
//! row **closes** with billed usage. Between "request dispatched" and
//! "attempt closed" — which is the entire visible lifetime of a long
//! streaming call — a host polling closed rows renders a gauge that is
//! stale or zero. Hosts need a truthful reading at every instant.
//!
//! ## Three-layer estimator
//!
//! Modeled on the reference estimator in oh-my-pi
//! (`session-stats.ts` `getContextBreakdown`), translated into the
//! durable worker's request/attempt lifecycle:
//!
//! 1. **Anchor** — the last settled call's billed `input_tokens`
//!    (inclusive of cache-read and cache-write slices, per
//!    [`agent_sdk_foundation::llm::Usage`] semantics). Exact, re-set
//!    at every attempt close, so the estimate converges to billed
//!    truth at every turn boundary.
//! 2. **Tail** — the bytes the request grew by since the anchor's
//!    dispatch, priced at [`ESTIMATE_BYTES_PER_TOKEN`] (the classic
//!    bytes/4 heuristic). Only the *delta* rides the heuristic; the
//!    anchored bulk stays exact.
//! 3. **Pending snapshot** — published at request dispatch, before
//!    the provider yields anything, and cleared when the call
//!    settles or aborts. While present it is the freshest truthful
//!    reading; afterwards the (new) anchor takes over.
//!
//! A request that *shrank* below the anchored request size means
//! history was rewritten (compaction / pruning); the estimator falls
//! back to a whole-request byte estimate instead of an anchor delta.
//!
//! ## Surfaces
//!
//! * [`estimate_chat_request`] / [`estimate_tokens_for_bytes`] — pure
//!   estimators hosts can also run over their own text (e.g. composer
//!   input that has not reached a request yet).
//! * [`ThreadEstimates`] — the per-thread estimator state machine.
//! * [`live`] — the process-wide registry. The worker publishes into
//!   it at [`dispatch`](ThreadEstimates::on_dispatch) /
//!   [`settle`](ThreadEstimates::on_settled) /
//!   [`abort`](ThreadEstimates::on_aborted); an embedding host reads
//!   [`ThreadEstimates::live_estimate`] keyed by thread id. Same
//!   process-global pattern as [`crate::observability`]'s metrics
//!   cache: hosts embed the worker in-process and need no dependency
//!   threading to consume it.
//!
//! Everything here is additive and display-oriented: no journal row,
//! no committed event, and no existing usage consumer changes shape.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::llm::{ChatRequest, Content, ContentBlock, Message, Usage};

/// Bytes of request content estimated per prompt token.
pub const ESTIMATE_BYTES_PER_TOKEN: u64 = 4;

/// Flat token estimate for one image block.
///
/// Vision inputs are billed by tile, not by payload bytes — a base64
/// body would grossly overestimate. ~1.6k tokens approximates a
/// full-size tile across the providers the SDK ships.
pub const IMAGE_BLOCK_ESTIMATE_TOKENS: u64 = 1_600;

/// Upper bound on tracked threads before the registry evicts the
/// least-recently-touched entry. Estimator state is ~5 machine words
/// per thread; the cap exists for hygiene, not memory pressure.
const MAX_TRACKED_THREADS: usize = 4_096;

/// Estimate tokens for a byte count at the bytes/4 heuristic.
#[must_use]
pub const fn estimate_tokens_for_bytes(bytes: u64) -> u64 {
    bytes.div_ceil(ESTIMATE_BYTES_PER_TOKEN)
}

/// A dispatched request's size reading: content bytes plus their
/// heuristic token estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestEstimate {
    /// Estimation-relevant content bytes of the request (image blocks
    /// contribute their flat token estimate re-expressed as bytes so
    /// byte deltas between consecutive requests stay meaningful).
    pub bytes: u64,
}

impl RequestEstimate {
    /// Heuristic prompt-token estimate for the whole request.
    #[must_use]
    pub const fn tokens(self) -> u64 {
        estimate_tokens_for_bytes(self.bytes)
    }
}

/// Walk a [`ChatRequest`] and produce its [`RequestEstimate`].
///
/// Counts the content that actually occupies provider context —
/// system prompt, message content, tool declarations — and skips
/// transport knobs (`max_tokens`, cache config, …). No allocation:
/// string lengths are summed in place.
#[must_use]
pub fn estimate_chat_request(request: &ChatRequest) -> RequestEstimate {
    let mut bytes = request.system.len() as u64;
    for message in &request.messages {
        bytes = bytes.saturating_add(message_bytes(message));
    }
    if let Some(tools) = &request.tools {
        for tool in tools {
            bytes = bytes
                .saturating_add(tool.name.len() as u64)
                .saturating_add(tool.description.len() as u64)
                .saturating_add(json_value_bytes(&tool.input_schema));
        }
    }
    RequestEstimate { bytes }
}

/// Per-message role framing overhead, in bytes.
const MESSAGE_OVERHEAD_BYTES: u64 = 8;

fn message_bytes(message: &Message) -> u64 {
    let content = match &message.content {
        Content::Text(text) => text.len() as u64,
        Content::Blocks(blocks) => blocks.iter().map(content_block_bytes).sum(),
    };
    content.saturating_add(MESSAGE_OVERHEAD_BYTES)
}

fn content_block_bytes(block: &ContentBlock) -> u64 {
    match block {
        ContentBlock::Text { text } => text.len() as u64,
        ContentBlock::Thinking {
            thinking,
            signature,
        } => (thinking.len() + signature.as_ref().map_or(0, String::len)) as u64,
        ContentBlock::RedactedThinking { data } => data.len() as u64,
        ContentBlock::OpaqueReasoning { provider, data } => {
            (provider.len() as u64).saturating_add(json_value_bytes(data))
        }
        ContentBlock::ToolUse {
            id, name, input, ..
        } => ((id.len() + name.len()) as u64).saturating_add(json_value_bytes(input)),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } => (tool_use_id.len() + content.len()) as u64,
        ContentBlock::Image { .. } => IMAGE_BLOCK_ESTIMATE_TOKENS * ESTIMATE_BYTES_PER_TOKEN,
        // Documents ride their payload size; providers bill extracted
        // text, so this is a coarse ceiling rather than a floor.
        ContentBlock::Document { source } => source.data.len() as u64,
        // Future block kinds default to zero contribution rather than
        // guessing; the anchor re-trues the estimate at the next close.
        _ => 0,
    }
}

/// Approximate serialized size of a JSON value without serializing it.
fn json_value_bytes(value: &serde_json::Value) -> u64 {
    match value {
        serde_json::Value::Null => 4,
        serde_json::Value::Bool(_) => 5,
        serde_json::Value::Number(_) => 8,
        serde_json::Value::String(text) => (text.len() as u64).saturating_add(2),
        serde_json::Value::Array(items) => items.iter().fold(2_u64, |acc, item| {
            acc.saturating_add(json_value_bytes(item)).saturating_add(1)
        }),
        serde_json::Value::Object(entries) => entries.iter().fold(2_u64, |acc, (key, item)| {
            acc.saturating_add(key.len() as u64 + 4)
                .saturating_add(json_value_bytes(item))
        }),
    }
}

/// Which estimator layer produced a [`LiveContextEstimate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatePhase {
    /// A request is in flight; the reading is the pending snapshot
    /// published at dispatch.
    InFlight,
    /// No request is in flight; the reading is the last settled
    /// call's billed prompt size — exact, not estimated.
    Anchored,
}

/// A truthful-at-this-instant prompt-occupancy reading for a thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveContextEstimate {
    /// Estimated (or, when [`EstimatePhase::Anchored`], billed) prompt
    /// tokens occupying the context window right now.
    pub prompt_tokens: u64,
    /// Which layer produced the reading.
    pub phase: EstimatePhase,
}

#[derive(Debug, Clone, Copy)]
struct Anchor {
    /// Billed prompt tokens (`Usage::input_tokens`, inclusive) of the
    /// last settled call.
    billed_prompt_tokens: u64,
    /// Request bytes of the dispatch that produced the anchor, when
    /// the settle could pair them with a pending snapshot.
    request_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct Pending {
    estimated_prompt_tokens: u64,
    request_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ThreadState {
    anchor: Option<Anchor>,
    pending: Option<Pending>,
    touch: u64,
}

/// Per-thread three-layer estimator states.
///
/// All operations are cheap constant-time map touches guarded by one
/// mutex; nothing here is awaited or held across I/O.
#[derive(Debug, Default)]
pub struct ThreadEstimates {
    inner: Mutex<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    threads: HashMap<String, ThreadState>,
    touch_counter: u64,
}

impl ThreadEstimates {
    /// Create an empty estimator registry (tests and embedded hosts;
    /// the worker publishes into [`live`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a request dispatch for `thread_id` and publish its
    /// pending snapshot. Returns the estimated prompt tokens.
    pub fn on_dispatch(&self, thread_id: &ThreadId, request: RequestEstimate) -> u64 {
        let mut inner = self.lock();
        inner.touch_counter += 1;
        let touch = inner.touch_counter;
        inner.evict_if_full(thread_id.0.as_str());
        let state = inner
            .threads
            .entry(thread_id.0.as_str().to_owned())
            .or_default();
        state.touch = touch;
        let estimated_prompt_tokens = match state.anchor {
            Some(Anchor {
                billed_prompt_tokens,
                request_bytes: Some(anchor_bytes),
            }) if request.bytes >= anchor_bytes => {
                // Anchor + tail: only the growth since the anchored
                // dispatch rides the bytes/4 heuristic.
                billed_prompt_tokens
                    .saturating_add(estimate_tokens_for_bytes(request.bytes - anchor_bytes))
            }
            Some(Anchor {
                billed_prompt_tokens,
                request_bytes: Some(_),
            }) => {
                // Request shrank below the anchored dispatch: history
                // was compacted/pruned, so the anchor no longer bounds
                // this request. Estimate the whole request; the next
                // settle re-anchors exactly. The stale anchor is NOT
                // kept as a floor — dropping is the honest reading.
                let _ = billed_prompt_tokens;
                request.tokens()
            }
            Some(Anchor {
                billed_prompt_tokens,
                request_bytes: None,
            }) => {
                // Anchor without paired bytes (settle arrived with no
                // pending snapshot, e.g. after a registry eviction).
                // The request can only have grown since that billed
                // call, so the anchor is a floor for the byte estimate.
                billed_prompt_tokens.max(request.tokens())
            }
            None => request.tokens(),
        };
        state.pending = Some(Pending {
            estimated_prompt_tokens,
            request_bytes: request.bytes,
        });
        drop(inner);
        estimated_prompt_tokens
    }

    /// Record a settled call for `thread_id`: clears the pending
    /// snapshot and re-anchors on billed usage.
    ///
    /// A settle with zero billed input tokens (providers that never
    /// reported usage) clears the snapshot but keeps the previous
    /// anchor — anchoring at zero would zero a gauge that was just
    /// showing a truthful estimate.
    pub fn on_settled(&self, thread_id: &ThreadId, usage: &Usage) {
        let mut inner = self.lock();
        inner.touch_counter += 1;
        let touch = inner.touch_counter;
        if let Some(state) = inner.threads.get_mut(thread_id.0.as_str()) {
            state.touch = touch;
            let pending = state.pending.take();
            if usage.input_tokens > 0 {
                state.anchor = Some(Anchor {
                    billed_prompt_tokens: u64::from(usage.input_tokens),
                    request_bytes: pending.map(|p| p.request_bytes),
                });
            }
        }
        drop(inner);
    }

    /// Record an aborted call (failure, cancellation, retry) for
    /// `thread_id`: clears the pending snapshot, keeps the anchor.
    pub fn on_aborted(&self, thread_id: &ThreadId) {
        let mut inner = self.lock();
        inner.touch_counter += 1;
        let touch = inner.touch_counter;
        if let Some(state) = inner.threads.get_mut(thread_id.0.as_str()) {
            state.touch = touch;
            state.pending = None;
        }
    }

    /// The current truthful occupancy reading for `thread_id`, or
    /// [`None`] when the estimator has never seen the thread.
    #[must_use]
    pub fn live_estimate(&self, thread_id: &ThreadId) -> Option<LiveContextEstimate> {
        let inner = self.lock();
        let reading = inner.threads.get(thread_id.0.as_str()).and_then(|state| {
            if let Some(pending) = state.pending {
                return Some(LiveContextEstimate {
                    prompt_tokens: pending.estimated_prompt_tokens,
                    phase: EstimatePhase::InFlight,
                });
            }
            state.anchor.map(|anchor| LiveContextEstimate {
                prompt_tokens: anchor.billed_prompt_tokens,
                phase: EstimatePhase::Anchored,
            })
        });
        drop(inner);
        reading
    }

    /// Drop all estimator state for `thread_id` (host hygiene hook).
    pub fn forget(&self, thread_id: &ThreadId) {
        self.lock().threads.remove(thread_id.0.as_str());
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // Estimator state is display-only; a panic while holding the
        // lock cannot corrupt anything worth failing over.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Inner {
    /// Evict the least-recently-touched thread when inserting a new
    /// key would exceed [`MAX_TRACKED_THREADS`].
    fn evict_if_full(&mut self, incoming: &str) {
        if self.threads.len() < MAX_TRACKED_THREADS || self.threads.contains_key(incoming) {
            return;
        }
        if let Some(oldest) = self
            .threads
            .iter()
            .min_by_key(|(_, state)| state.touch)
            .map(|(key, _)| key.clone())
        {
            self.threads.remove(&oldest);
        }
    }
}

/// The process-wide live-estimate registry.
///
/// The durable worker publishes every dispatched root-turn LLM call
/// here; hosts that embed the worker in-process read it keyed by
/// thread id to drive live gauges between attempt closes.
#[must_use]
pub fn live() -> &'static ThreadEstimates {
    static LIVE: LazyLock<ThreadEstimates> = LazyLock::new(ThreadEstimates::default);
    &LIVE
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::llm::Tool;
    use agent_sdk_foundation::types::ToolTier;

    fn thread(name: &str) -> ThreadId {
        ThreadId::from_string(format!("thread_ctx_est_{name}"))
    }

    fn usage(input_tokens: u32) -> Usage {
        Usage {
            input_tokens,
            output_tokens: 64,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            served_speed: None,
        }
    }

    fn request_of_bytes(bytes: u64) -> RequestEstimate {
        RequestEstimate { bytes }
    }

    /// Relative error of `estimate` against `billed`.
    fn relative_error(estimate: u64, billed: u64) -> f64 {
        assert!(billed > 0, "billed baseline must be positive");
        #[allow(clippy::cast_precision_loss)]
        {
            (estimate as f64 - billed as f64).abs() / billed as f64
        }
    }

    // ── Contract: |estimate − billed| relative error bounds ─────────

    /// Anchored state, nothing in flight: the reading IS the billed
    /// figure — zero error at every turn boundary (convergence).
    #[test]
    fn anchored_reading_matches_billed_exactly() {
        let estimates = ThreadEstimates::new();
        let id = thread("anchor_exact");

        estimates.on_dispatch(&id, request_of_bytes(400_000));
        estimates.on_settled(&id, &usage(97_000));

        let reading = estimates.live_estimate(&id).expect("anchored reading");
        assert_eq!(reading.phase, EstimatePhase::Anchored);
        assert_eq!(reading.prompt_tokens, 97_000);
    }

    /// Pending state with an anchor: only the tail rides bytes/4, so
    /// even when the true tokenizer density spans 3–5 bytes/token the
    /// relative error against the eventually-billed prompt stays
    /// under 5% for a tail that grew the request by ~10%.
    #[test]
    fn pending_with_anchor_bounds_relative_error_across_densities() {
        for true_bytes_per_token in [3_u64, 4, 5] {
            let estimates = ThreadEstimates::new();
            let id = thread("anchor_tail");

            // Call 1: anchor at billed truth.
            estimates.on_dispatch(&id, request_of_bytes(400_000));
            let billed_anchor = 100_000_u64;
            estimates.on_settled(&id, &usage(u32::try_from(billed_anchor).expect("fits")));

            // Call 2: request grew by 40_000 bytes of new content.
            let tail_bytes = 40_000_u64;
            let published = estimates.on_dispatch(&id, request_of_bytes(400_000 + tail_bytes));
            let reading = estimates.live_estimate(&id).expect("pending reading");
            assert_eq!(reading.phase, EstimatePhase::InFlight);
            assert_eq!(reading.prompt_tokens, published);

            // What the provider will bill: anchor + the tail at its
            // true density.
            let billed_next = billed_anchor + tail_bytes / true_bytes_per_token;
            assert!(
                relative_error(published, billed_next) < 0.05,
                "density {true_bytes_per_token}B/tok: estimate {published} vs billed {billed_next}",
            );
        }
    }

    /// Pending state with no anchor (first-ever call): the whole
    /// request rides bytes/4, so the error is bounded by tokenizer
    /// density variance — and collapses to zero once the call settles.
    #[test]
    fn pending_without_anchor_bounds_error_then_converges() {
        for true_bytes_per_token in [3_u64, 4, 5] {
            let estimates = ThreadEstimates::new();
            let id = thread("cold_start");

            let request_bytes = 120_000_u64;
            let published = estimates.on_dispatch(&id, request_of_bytes(request_bytes));
            let billed = request_bytes / true_bytes_per_token;
            assert!(
                relative_error(published, billed) <= 1.0 / 3.0 + f64::EPSILON,
                "density {true_bytes_per_token}B/tok: estimate {published} vs billed {billed}",
            );

            estimates.on_settled(&id, &usage(u32::try_from(billed).expect("fits")));
            let reading = estimates.live_estimate(&id).expect("anchored reading");
            assert_eq!(reading.prompt_tokens, billed, "converged to billed truth");
            assert_eq!(reading.phase, EstimatePhase::Anchored);
        }
    }

    // ── Pending-snapshot lifecycle semantics ─────────────────────────

    #[test]
    fn dispatch_publishes_immediately_before_any_provider_frame() {
        let estimates = ThreadEstimates::new();
        let id = thread("immediate");

        assert!(estimates.live_estimate(&id).is_none(), "unseen thread");
        let published = estimates.on_dispatch(&id, request_of_bytes(8_000));
        assert_eq!(published, 2_000);
        let reading = estimates.live_estimate(&id).expect("in-flight reading");
        assert_eq!(reading.phase, EstimatePhase::InFlight);
        assert_eq!(reading.prompt_tokens, 2_000);
    }

    #[test]
    fn abort_clears_pending_and_falls_back_to_anchor() {
        let estimates = ThreadEstimates::new();
        let id = thread("abort");

        estimates.on_dispatch(&id, request_of_bytes(40_000));
        estimates.on_settled(&id, &usage(11_000));
        estimates.on_dispatch(&id, request_of_bytes(48_000));
        estimates.on_aborted(&id);

        let reading = estimates.live_estimate(&id).expect("anchored fallback");
        assert_eq!(reading.phase, EstimatePhase::Anchored);
        assert_eq!(reading.prompt_tokens, 11_000);
    }

    #[test]
    fn abort_without_anchor_returns_no_reading() {
        let estimates = ThreadEstimates::new();
        let id = thread("abort_cold");

        estimates.on_dispatch(&id, request_of_bytes(4_000));
        estimates.on_aborted(&id);
        assert!(estimates.live_estimate(&id).is_none());
    }

    #[test]
    fn zero_usage_settle_keeps_previous_anchor() {
        let estimates = ThreadEstimates::new();
        let id = thread("zero_settle");

        estimates.on_dispatch(&id, request_of_bytes(40_000));
        estimates.on_settled(&id, &usage(9_000));
        estimates.on_dispatch(&id, request_of_bytes(44_000));
        estimates.on_settled(&id, &usage(0));

        let reading = estimates.live_estimate(&id).expect("anchor survives");
        assert_eq!(reading.phase, EstimatePhase::Anchored);
        assert_eq!(reading.prompt_tokens, 9_000);
    }

    #[test]
    fn shrunken_request_re_estimates_instead_of_anchoring_high() {
        let estimates = ThreadEstimates::new();
        let id = thread("compaction");

        estimates.on_dispatch(&id, request_of_bytes(800_000));
        estimates.on_settled(&id, &usage(210_000));

        // Compaction rewrote history: the next request is far smaller.
        let published = estimates.on_dispatch(&id, request_of_bytes(100_000));
        assert_eq!(
            published, 25_000,
            "a compacted request must not keep the stale anchor as a floor",
        );
    }

    #[test]
    fn threads_are_independent() {
        let estimates = ThreadEstimates::new();
        let a = thread("indep_a");
        let b = thread("indep_b");

        estimates.on_dispatch(&a, request_of_bytes(8_000));
        estimates.on_dispatch(&b, request_of_bytes(80_000));
        estimates.on_settled(&b, &usage(19_000));

        assert_eq!(
            estimates.live_estimate(&a).map(|r| r.phase),
            Some(EstimatePhase::InFlight),
        );
        assert_eq!(
            estimates.live_estimate(&b).map(|r| r.prompt_tokens),
            Some(19_000),
        );
    }

    #[test]
    fn forget_drops_state() {
        let estimates = ThreadEstimates::new();
        let id = thread("forget");
        estimates.on_dispatch(&id, request_of_bytes(8_000));
        estimates.forget(&id);
        assert!(estimates.live_estimate(&id).is_none());
    }

    #[test]
    fn registry_evicts_least_recently_touched_at_capacity() {
        let estimates = ThreadEstimates::new();
        let first = thread("evict_first");
        estimates.on_dispatch(&first, request_of_bytes(4_000));
        for index in 1..MAX_TRACKED_THREADS {
            let filler = ThreadId::from_string(format!("thread_ctx_est_fill_{index}"));
            estimates.on_dispatch(&filler, request_of_bytes(4_000));
        }
        let overflow = thread("evict_overflow");
        estimates.on_dispatch(&overflow, request_of_bytes(4_000));

        assert!(
            estimates.live_estimate(&first).is_none(),
            "oldest entry evicted at capacity",
        );
        assert!(estimates.live_estimate(&overflow).is_some());
    }

    // ── Request walker ───────────────────────────────────────────────

    #[test]
    fn chat_request_estimate_counts_system_messages_and_tools() {
        let system = "s".repeat(4_000);
        let request =
            ChatRequest::new(system, vec![Message::user("u".repeat(2_000))]).with_tools(vec![
                Tool {
                    name: "read".to_owned(),
                    description: "d".repeat(400),
                    input_schema: serde_json::json!({"type": "object"}),
                    display_name: "Read".to_owned(),
                    tier: ToolTier::Observe,
                },
            ]);

        let bare = estimate_chat_request(&ChatRequest::new("", Vec::new()));
        let full = estimate_chat_request(&request);
        assert_eq!(bare.bytes, 0);
        assert!(full.bytes > 6_400, "system + message + tool all counted");
        assert!(full.tokens() >= full.bytes / ESTIMATE_BYTES_PER_TOKEN);
    }

    #[test]
    fn image_blocks_cost_flat_tokens_not_payload_bytes() {
        let payload = "A".repeat(1_000_000);
        let request = ChatRequest::new(
            "",
            vec![Message::user_with_content(vec![ContentBlock::Image {
                source: agent_sdk_foundation::llm::ContentSource::new("image/png", payload),
            }])],
        );
        let estimate = estimate_chat_request(&request);
        assert_eq!(
            estimate.tokens(),
            IMAGE_BLOCK_ESTIMATE_TOKENS + MESSAGE_OVERHEAD_BYTES.div_ceil(ESTIMATE_BYTES_PER_TOKEN),
        );
    }

    #[test]
    fn estimate_is_monotone_in_content() {
        let small = ChatRequest::new("sys", vec![Message::user("short")]);
        let large = ChatRequest::new(
            "sys",
            vec![
                Message::user("short"),
                Message::assistant("a much longer reply with tool traffic"),
                Message::tool_result("tc_1", "x".repeat(10_000), false),
            ],
        );
        assert!(estimate_chat_request(&large).bytes > estimate_chat_request(&small).bytes);
    }
}
