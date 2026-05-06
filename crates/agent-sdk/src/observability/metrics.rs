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
//! first call to [`Metrics::global`] / [`Metrics::init`]. Tests that
//! swap in a fresh `opentelemetry_sdk::metrics::SdkMeterProvider`
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
/// Streaming TTFC / TPOC instruments are intentionally **not**
/// declared here. They are added by Track B2 alongside the per-chunk
/// timing recorder so the bound to the meter provider happens at the
/// same site as the recorder — declaring them ahead of time would
/// otherwise produce dead-code warnings until B2 lands.
#[derive(Debug)]
pub struct Metrics {
    pub(crate) token_usage: Histogram<u64>,
    pub(crate) operation_duration: Histogram<f64>,

    pub(crate) turns_duration: Histogram<f64>,
    pub(crate) runs_outcome: Counter<u64>,
    pub(crate) tools_execution_duration: Histogram<f64>,
    pub(crate) tools_execution_count: Counter<u64>,
    pub(crate) context_compaction: Counter<u64>,
    pub(crate) context_compaction_tokens_saved: Histogram<u64>,
    pub(crate) subagent_invocations: Counter<u64>,
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

    /// Drop the cached singleton.
    ///
    /// Test-only escape hatch. Production code never calls this — the
    /// instruments are bound to whichever meter provider was current
    /// when [`init`](Self::init) ran, and rebuilding them against a
    /// different provider mid-run would silently lose data points.
    pub fn reset_for_testing() {
        let mut guard = match METRICS.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = None;
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
            turns_duration,
            runs_outcome,
            tools_execution_duration,
            tools_execution_count,
            context_compaction,
            context_compaction_tokens_saved,
            subagent_invocations,
            mcp_requests_duration,
            llm_retries,
        }
    }
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
