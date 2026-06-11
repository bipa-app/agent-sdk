//! `GenAI` client + SDK-specific metric instruments.
//!
//! Compiled only with `feature = "otel"`. The module owns a small,
//! lazily-initialised [`Metrics`] singleton that the rest of the SDK
//! reaches via [`Metrics::global`]. Every histogram declared here uses
//! buckets defined by the `OpenTelemetry` `GenAI` metrics semantic
//! conventions; we never accept the SDK default boundaries because
//! they are tuned for HTTP timings, not `GenAI` ranges.
//!
//! ## Lifecycle
//!
//! Instruments are bound to whichever meter provider is current at the
//! first call to [`Metrics::global`] / [`Metrics::init`].
//! `agent_sdk_otel::install_global_provider` calls [`Metrics::rebind`]
//! right after installing the global meter provider so the singleton binds
//! to the real provider even if an earlier code path lazily built it
//! against the no-op meter (or a previous provider was replaced). Tests
//! that swap in a fresh `opentelemetry_sdk::metrics::SdkMeterProvider`
//! between cases must call [`Metrics::reset_for_testing`] so the next
//! lookup rebuilds the cache against the new provider.
//!
//! ## Naming
//!
//! * `gen_ai.client.*` — `OTel` `GenAI` client metrics ([spec][spec]).
//! * `agent_sdk.*` — SDK-specific instruments. The namespace is shared
//!   with the span attributes in [`super::attrs`] so dashboards and
//!   alerts can correlate a metric and a span without translation.
//!
//! [spec]: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/

use std::sync::{Arc, RwLock};

use opentelemetry::global;
use opentelemetry::metrics::{Counter, Histogram};

/// Bucket boundaries for `gen_ai.client.token.usage` (`{token}`).
///
/// Mirrors the spec's recommended `ExplicitBucketBoundaries`.
const TOKEN_USAGE_BUCKETS: &[f64] = &[
    1.0,
    4.0,
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
];

/// Bucket boundaries for `gen_ai.client.operation.duration` and the
/// streaming TTFC / TPOC histograms (`s`).
///
/// Also reused by `agent_sdk.turns.duration` and
/// `agent_sdk.mcp.requests.duration` because their typical ranges
/// overlap LLM call latencies.
const SHORT_DURATION_BUCKETS_S: &[f64] = &[
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];

/// Bucket boundaries for `agent_sdk.tools.execution.duration` (`ms`).
///
/// Tool execution mixes near-instant `Observe` reads with multi-minute
/// async work, so we span 1 ms .. 5 minutes with denser buckets at the
/// short end where most calls land.
const TOOL_DURATION_BUCKETS_MS: &[f64] = &[
    1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1_000.0, 5_000.0, 10_000.0, 60_000.0, 300_000.0,
];

/// Container for every metric instrument the SDK records.
///
/// The struct is held behind an `Arc` so call sites that need
/// frequent recording (per LLM call, per tool call) can clone the
/// handle into an async future without re-walking a `Mutex`.
///
/// The streaming TTFC / TPOC histograms live alongside the
/// non-streaming `operation_duration` because the recorder
/// (`agent_loop::llm::process_stream`) treats them as paired
/// instruments — one fires once per stream, the other once per
/// post-first chunk.
#[derive(Debug)]
pub struct Metrics {
    pub(crate) token_usage: Histogram<u64>,
    pub(crate) operation_duration: Histogram<f64>,
    pub(crate) time_to_first_chunk: Histogram<f64>,
    pub(crate) time_per_output_chunk: Histogram<f64>,

    pub(crate) turns_duration: Histogram<f64>,
    pub(crate) runs_outcome: Counter<u64>,
    pub(crate) tools_execution_duration: Histogram<f64>,
    pub(crate) tools_execution_count: Counter<u64>,
    pub(crate) context_compaction: Counter<u64>,
    pub(crate) context_compaction_tokens_saved: Histogram<u64>,
    pub(crate) subagent_invocations: Counter<u64>,
    // Only recorded by the MCP client, which is compiled behind the `mcp`
    // feature; gated here so the field is not dead code without it.
    #[cfg(feature = "mcp")]
    pub(crate) mcp_requests_duration: Histogram<f64>,
    pub(crate) llm_retries: Counter<u64>,
}

static METRICS: RwLock<Option<Arc<Metrics>>> = RwLock::new(None);

impl Metrics {
    /// Build instruments under the supplied meter scope and cache the
    /// resulting [`Arc<Metrics>`] for subsequent [`Metrics::global`]
    /// callers.
    ///
    /// If the cache is already populated the supplied `name` is
    /// ignored and the cached handle is returned — the first caller
    /// in the process wins. Tests that rotate the global meter
    /// provider between cases must call
    /// [`Metrics::reset_for_testing`] beforehand so the next `init`
    /// rebuilds against the fresh provider.
    #[must_use]
    pub fn init(name: &'static str) -> Arc<Self> {
        if let Some(existing) = read_cached() {
            return existing;
        }

        let built = Arc::new(Self::build(name));
        write_cached(Arc::clone(&built));
        built
    }

    /// Convenience wrapper that initialises the singleton under the
    /// `agent-sdk` meter scope.
    #[must_use]
    pub fn global() -> Arc<Self> {
        Self::init(env!("CARGO_PKG_NAME"))
    }

    /// Drop the cached instrument singleton so the next
    /// [`Metrics::global`] call rebuilds against the **currently
    /// installed** global meter provider.
    ///
    /// `agent_sdk_otel::install_global_provider` calls this immediately
    /// after `global::set_meter_provider`. Without it, any telemetry path
    /// that lazily built the singleton before the provider was installed
    /// (or before a re-install) would stay bound to the no-op meter — or a
    /// now-shut-down provider — for the rest of the process, silently
    /// dropping every counter / histogram. Calling it at install time is
    /// safe because no real data points exist yet; it must NOT be used
    /// mid-run, where rebuilding would lose in-flight aggregation.
    pub fn rebind() {
        clear_cache();
    }

    /// Drop the cached singleton.
    ///
    /// Test-only escape hatch so tests that rotate the global meter
    /// provider between cases force a rebuild against the fresh provider.
    pub fn reset_for_testing() {
        clear_cache();
    }

    /// Record the `gen_ai.client.token.usage` histogram for a chat
    /// response, splitting one data point per non-zero token type
    /// (`input` / `output` / `cache_read` / `cache_creation`).
    ///
    /// This is the single source of truth for the token-usage label
    /// set so the in-process `agent_loop` and the
    /// daemon-hosted `agent-server` worker emit byte-identical labels.
    /// Splitting by type keeps the histogram aggregatable in
    /// Prometheus / Grafana — collapsing the four types into one
    /// record would erase the cache-hit-ratio dimension dashboards
    /// care about most.
    pub fn record_chat_token_usage(
        &self,
        usage: &crate::llm::Usage,
        provider_name: &'static str,
        request_model: &str,
        response_model: &str,
    ) {
        use opentelemetry::KeyValue;

        let entries: [(u32, &'static str); 4] = [
            (usage.input_tokens, "input"),
            (usage.output_tokens, "output"),
            (usage.cached_input_tokens, "cache_read"),
            (usage.cache_creation_input_tokens, "cache_creation"),
        ];

        for (count, token_type) in entries {
            if count == 0 {
                continue;
            }
            self.token_usage.record(
                u64::from(count),
                &[
                    KeyValue::new(super::attrs::GEN_AI_OPERATION_NAME, "chat"),
                    KeyValue::new(super::attrs::GEN_AI_PROVIDER_NAME, provider_name),
                    KeyValue::new("gen_ai.token.type", token_type),
                    KeyValue::new(
                        super::attrs::GEN_AI_REQUEST_MODEL,
                        request_model.to_string(),
                    ),
                    KeyValue::new(
                        super::attrs::GEN_AI_RESPONSE_MODEL,
                        response_model.to_string(),
                    ),
                ],
            );
        }
    }

    /// Record a `gen_ai.client.operation.duration` sample for a
    /// successful chat call. The label set mirrors the success arm of
    /// the in-process loop so both code paths land in the same series.
    pub fn record_chat_operation_duration_success(
        &self,
        elapsed_secs: f64,
        provider_name: &'static str,
        request_model: &str,
        response_model: &str,
    ) {
        use opentelemetry::KeyValue;

        self.operation_duration.record(
            elapsed_secs,
            &[
                KeyValue::new(super::attrs::GEN_AI_OPERATION_NAME, "chat"),
                KeyValue::new(super::attrs::GEN_AI_PROVIDER_NAME, provider_name),
                KeyValue::new(
                    super::attrs::GEN_AI_REQUEST_MODEL,
                    request_model.to_string(),
                ),
                KeyValue::new(
                    super::attrs::GEN_AI_RESPONSE_MODEL,
                    response_model.to_string(),
                ),
            ],
        );
    }

    /// Record a `gen_ai.client.operation.duration` sample for a failed
    /// chat call, carrying the stable `error.type` label in place of
    /// the response model. Mirrors the error arm of the in-process
    /// loop.
    pub fn record_chat_operation_duration_error(
        &self,
        elapsed_secs: f64,
        provider_name: &'static str,
        request_model: &str,
        error_type: &'static str,
    ) {
        use opentelemetry::KeyValue;

        self.operation_duration.record(
            elapsed_secs,
            &[
                KeyValue::new(super::attrs::GEN_AI_OPERATION_NAME, "chat"),
                KeyValue::new(super::attrs::GEN_AI_PROVIDER_NAME, provider_name),
                KeyValue::new(
                    super::attrs::GEN_AI_REQUEST_MODEL,
                    request_model.to_string(),
                ),
                KeyValue::new(super::attrs::ERROR_TYPE, error_type),
            ],
        );
    }

    /// Record the `agent_sdk.tools.execution.count` counter and, when a
    /// duration is known, the `agent_sdk.tools.execution.duration`
    /// histogram for a single tool invocation.
    ///
    /// `outcome` is one of the stable strings emitted by the loop
    /// (`success` / `error` / `blocked` / `rejected` /
    /// `awaiting_confirmation`). Both instruments share the same three
    /// labels (`gen_ai.tool.name`, `agent_sdk.tool.kind`,
    /// `agent_sdk.tool.outcome`) so a dashboard can join the count and
    /// the duration without translation.
    pub fn record_tool_execution(
        &self,
        tool_name: &str,
        tool_kind: &'static str,
        outcome: &'static str,
        duration_ms: Option<u64>,
    ) {
        use opentelemetry::KeyValue;

        let metric_attrs = [
            KeyValue::new(super::attrs::GEN_AI_TOOL_NAME, tool_name.to_string()),
            KeyValue::new(super::attrs::SDK_TOOL_KIND, tool_kind),
            KeyValue::new(super::attrs::SDK_TOOL_OUTCOME, outcome),
        ];
        self.tools_execution_count.add(1, &metric_attrs);
        if let Some(ms) = duration_ms {
            self.tools_execution_duration
                .record(tool_duration_ms_to_f64(ms), &metric_attrs);
        }
    }

    fn build(scope: &'static str) -> Self {
        let meter = global::meter(scope);

        let token_usage = meter
            .u64_histogram("gen_ai.client.token.usage")
            .with_unit("{token}")
            .with_description("Number of input and output tokens used.")
            .with_boundaries(TOKEN_USAGE_BUCKETS.to_vec())
            .build();
        let operation_duration = meter
            .f64_histogram("gen_ai.client.operation.duration")
            .with_unit("s")
            .with_description("GenAI operation duration.")
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();
        let time_to_first_chunk = meter
            .f64_histogram("gen_ai.client.operation.time_to_first_chunk")
            .with_unit("s")
            .with_description("Time to first response chunk for streaming GenAI operations.")
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();
        let time_per_output_chunk = meter
            .f64_histogram("gen_ai.client.operation.time_per_output_chunk")
            .with_unit("s")
            .with_description(
                "Time per output chunk after the first chunk for streaming GenAI operations.",
            )
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();

        let turns_duration = meter
            .f64_histogram("agent_sdk.turns.duration")
            .with_unit("s")
            .with_description("Wall-clock duration of a single agent turn.")
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();
        let runs_outcome = meter
            .u64_counter("agent_sdk.runs.outcome")
            .with_description("Count of completed agent runs by outcome.")
            .build();
        let tools_execution_duration = meter
            .f64_histogram("agent_sdk.tools.execution.duration")
            .with_unit("ms")
            .with_description("Duration of a single tool invocation.")
            .with_boundaries(TOOL_DURATION_BUCKETS_MS.to_vec())
            .build();
        let tools_execution_count = meter
            .u64_counter("agent_sdk.tools.execution.count")
            .with_description("Count of tool invocations by name, kind, and outcome.")
            .build();
        let context_compaction = meter
            .u64_counter("agent_sdk.context.compaction")
            .with_description("Count of context-compaction operations by trigger.")
            .build();
        let context_compaction_tokens_saved = meter
            .u64_histogram("agent_sdk.context.compaction.tokens_saved")
            .with_unit("{token}")
            .with_description("Tokens saved by a single compaction operation.")
            .with_boundaries(TOKEN_USAGE_BUCKETS.to_vec())
            .build();
        let subagent_invocations = meter
            .u64_counter("agent_sdk.subagent.invocations")
            .with_description("Count of subagent invocations by agent name and outcome.")
            .build();
        #[cfg(feature = "mcp")]
        let mcp_requests_duration = meter
            .f64_histogram("agent_sdk.mcp.requests.duration")
            .with_unit("s")
            .with_description("Duration of an MCP JSON-RPC client request.")
            .with_boundaries(SHORT_DURATION_BUCKETS_S.to_vec())
            .build();
        let llm_retries = meter
            .u64_counter("agent_sdk.llm.retries")
            .with_description("Count of LLM call retries by provider and error type.")
            .build();

        Self {
            token_usage,
            operation_duration,
            time_to_first_chunk,
            time_per_output_chunk,
            turns_duration,
            runs_outcome,
            tools_execution_duration,
            tools_execution_count,
            context_compaction,
            context_compaction_tokens_saved,
            subagent_invocations,
            #[cfg(feature = "mcp")]
            mcp_requests_duration,
            llm_retries,
        }
    }
}

/// Convert a `ToolResult::duration_ms` value (`u64` milliseconds)
/// into a histogram-friendly `f64`.
///
/// Bounding through `u32` keeps the conversion lossless because
/// `u32::MAX` ≈ 49.7 days — far above the histogram's 5-minute top
/// bucket, so any clamped value still falls in the overflow bucket
/// dashboards expect. The clamp path also emits a `warn!` so a real
/// runaway duration is investigable rather than silently swallowed.
#[must_use]
pub fn tool_duration_ms_to_f64(ms: u64) -> f64 {
    if let Ok(v) = u32::try_from(ms) {
        return f64::from(v);
    }
    log::warn!("tool duration {ms}ms exceeds u32::MAX; clamping for histogram");
    f64::from(u32::MAX)
}

fn clear_cache() {
    let mut guard = match METRICS.write() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    *guard = None;
}

fn read_cached() -> Option<Arc<Metrics>> {
    let guard = match METRICS.read() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.as_ref().map(Arc::clone)
}

fn write_cached(value: Arc<Metrics>) {
    let mut guard = match METRICS.write() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.is_none() {
        *guard = Some(value);
    }
    // If a concurrent caller raced us in, drop the freshly-built
    // handle on the floor. Both copies are functionally equivalent
    // (same scope, same global meter), so the duplicate is harmless.
}

#[cfg(test)]
mod tests {
    use super::Metrics;

    #[test]
    fn rebind_forces_fresh_instruments_against_current_provider() {
        // Populate the cache, then rebind: the next `global()` must rebuild
        // a distinct handle bound to whatever provider is now installed,
        // rather than returning the stale (possibly no-op-bound) singleton.
        let first = Metrics::global();
        Metrics::rebind();
        let second = Metrics::global();
        assert!(
            !std::sync::Arc::ptr_eq(&first, &second),
            "rebind must clear the cache so the next global() rebuilds"
        );
    }
}
