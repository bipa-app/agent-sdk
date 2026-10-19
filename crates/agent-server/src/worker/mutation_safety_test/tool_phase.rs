//! A tool call's store-side phases (dogfood D15): each is an
//! `agent_server.tool.phase` span carrying the call's ids, nested under the
//! tool task's span, and a sample on `agent_server.tool.phase.duration`
//! labelled only by phase and outcome.

use std::collections::BTreeSet;

use opentelemetry::trace::{
    Span as _, SpanContext, TraceContextExt as _, Tracer as _, TracerProvider as _,
};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};

use super::*;
use crate::observability::attrs;

const SPAN: &str = "agent_server.tool.phase";
const HISTOGRAM: &str = "agent_server.tool.phase.duration";

struct Exporters {
    spans: InMemorySpanExporter,
    tracer: SdkTracerProvider,
    metrics: InMemoryMetricExporter,
    meter: SdkMeterProvider,
}

impl Exporters {
    fn install() -> Self {
        let metrics = InMemoryMetricExporter::default();
        let meter = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(metrics.clone()).build())
            .build();
        opentelemetry::global::set_meter_provider(meter.clone());
        let spans = InMemorySpanExporter::default();
        let tracer = SdkTracerProvider::builder()
            .with_simple_exporter(spans.clone())
            .build();
        opentelemetry::global::set_tracer_provider(tracer.clone());
        opentelemetry::global::set_text_map_propagator(
            opentelemetry_sdk::propagation::TraceContextPropagator::new(),
        );
        crate::observability::ServerMetrics::reset_for_testing();
        Self {
            spans,
            tracer,
            metrics,
            meter,
        }
    }

    /// The span the host stores on a tool child as its `traceparent`.
    fn task_span(&self) -> (SpanContext, String) {
        let mut span = self.tracer.tracer("test").start("tool task");
        let context = span.span_context().clone();
        span.end();
        let traceparent = format!("00-{}-{}-01", context.trace_id(), context.span_id());
        (context, traceparent)
    }

    /// The phase spans of `task`, in the order they ended.
    fn phases(&self, task: &AgentTaskId) -> Result<Vec<SpanData>> {
        self.tracer.force_flush()?;
        Ok(self
            .spans
            .get_finished_spans()?
            .into_iter()
            .filter(|span| {
                span.name == SPAN
                    && attr(span, attrs::SPAN_TASK_ID).as_deref() == Some(task.as_str())
            })
            .collect())
    }

    /// Every finished span carrying `agent_server.task.id` = `task`, the
    /// one task key the SDK and bip (after bip #985) both set.
    fn task_spans(&self, task: &AgentTaskId) -> Result<Vec<SpanData>> {
        self.tracer.force_flush()?;
        Ok(self
            .spans
            .get_finished_spans()?
            .into_iter()
            .filter(|span| attr(span, "agent_server.task.id").as_deref() == Some(task.as_str()))
            .collect())
    }

    /// Every data point of the phase histogram, as its label set.
    fn histogram_labels(&self) -> Result<Vec<BTreeSet<(String, String)>>> {
        self.meter.force_flush()?;
        let exported = self.metrics.get_finished_metrics()?;
        let latest = exported.last().context("no metric export")?;
        Ok(latest
            .scope_metrics()
            .flat_map(opentelemetry_sdk::metrics::data::ScopeMetrics::metrics)
            .filter(|metric| metric.name() == HISTOGRAM)
            .flat_map(|metric| match metric.data() {
                AggregatedMetrics::F64(MetricData::Histogram(histogram)) => histogram
                    .data_points()
                    .map(|point| {
                        point
                            .attributes()
                            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
                            .collect()
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect())
    }
}

fn attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.to_string())
}

/// Assert `spans` are exactly `expected` (phase, outcome), each with the
/// call's ids and nested under `parent`.
fn assert_phases(
    spans: &[SpanData],
    expected: &[(&str, &str)],
    ctx: &ToolTaskBootstrap,
    parent: &SpanContext,
) {
    let got: Vec<(Option<String>, Option<String>)> = spans
        .iter()
        .map(|span| {
            (
                attr(span, attrs::SPAN_TOOL_PHASE),
                attr(span, attrs::OUTCOME),
            )
        })
        .collect();
    let want: Vec<(Option<String>, Option<String>)> = expected
        .iter()
        .map(|(phase, outcome)| (Some((*phase).to_owned()), Some((*outcome).to_owned())))
        .collect();
    assert_eq!(got, want, "phase spans (phase, outcome)");
    for span in spans {
        assert_eq!(
            attr(span, "gen_ai.tool.call.id").as_deref(),
            Some(ctx.tool_call.id.as_str())
        );
        assert_eq!(
            attr(span, "gen_ai.tool.name").as_deref(),
            Some(ctx.tool_call.name.as_str())
        );
        assert_eq!(span.span_context.trace_id(), parent.trace_id());
        assert_eq!(
            span.parent_span_id,
            parent.span_id(),
            "the phase nests under the tool task's span"
        );
    }
}

/// The bootstrap of a fresh tool child running `tool`, carrying
/// `traceparent` as the host stores it.
async fn tool_child(
    stores: &TestStores,
    tool: &str,
    traceparent: Option<String>,
) -> Result<ToolTaskBootstrap> {
    let (_parent, children) = suspend_root_with_tools(
        stores,
        vec![("call_phase".into(), tool.into(), serde_json::json!({}))],
    )
    .await?;
    let child = acquire_child(&stores.tasks, &children[0].id).await?;
    let mut ctx = resolve_tool_bootstrap(child, &stores.tasks, ActivityBeacon::default()).await?;
    ctx.child_task.otel_traceparent = traceparent;
    Ok(ctx)
}

#[tokio::test]
async fn an_unsafe_call_times_its_lookup_claim_and_settlement() -> Result<()> {
    let otel = Exporters::install();
    let (parent, traceparent) = otel.task_span();
    let stores = TestStores::new();
    let ctx = tool_child(&stores, "bash", Some(traceparent)).await?;

    let outcome = run_tool_effect(
        ctx.clone(),
        &stores.tasks,
        &CancellationToken::new(),
        |_, _collector| async { Ok(ToolResult::success("ran")) },
        t_plus(20),
    )
    .await?;
    assert!(matches!(outcome, ToolTaskOutcome::Completed { .. }));

    assert_phases(
        &otel.phases(&ctx.task_id)?,
        &[
            ("lookup", "success"),
            ("claim", "success"),
            ("settlement", "success"),
        ],
        &ctx,
        &parent,
    );
    let labels = otel.histogram_labels()?;
    let phases: BTreeSet<(String, String)> = ["lookup", "claim", "settlement"]
        .into_iter()
        .map(|phase| ("phase".to_owned(), phase.to_owned()))
        .collect();
    for phase in &phases {
        assert!(
            labels.iter().any(|set| set.contains(phase)),
            "{HISTOGRAM} has a {phase:?} sample: {labels:?}"
        );
    }
    for set in &labels {
        let keys: Vec<&str> = set.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, ["outcome", "phase"], "bounded labels only: {set:?}");
    }
    Ok(())
}

/// A replay-safe call (a `read`) has no effect record, so its only store
/// phase is the settle.
#[tokio::test]
async fn a_safe_call_times_only_its_settlement() -> Result<()> {
    let otel = Exporters::install();
    let (parent, traceparent) = otel.task_span();
    let stores = TestStores::new();
    let ctx = tool_child(&stores, "read", Some(traceparent)).await?;

    let outcome = run_tool_effect(
        ctx.clone(),
        &stores.tasks,
        &CancellationToken::new(),
        |_, _collector| async { Ok(ToolResult::success("contents")) },
        t_plus(20),
    )
    .await?;
    assert!(matches!(outcome, ToolTaskOutcome::Completed { .. }));

    assert_phases(
        &otel.phases(&ctx.task_id)?,
        &[("settlement", "success")],
        &ctx,
        &parent,
    );
    Ok(())
}

/// An unsafe call found started is settled as interrupted without a claim:
/// its phases are the lookup that found the record and that settle.
#[tokio::test]
async fn an_interrupted_unsafe_replay_times_its_lookup_and_settlement() -> Result<()> {
    let otel = Exporters::install();
    let (parent, traceparent) = otel.task_span();
    let stores = TestStores::new();
    let ctx = tool_child(&stores, "bash", Some(traceparent)).await?;
    seed_effect(&stores.tasks, &ctx, ToolReplay::Unsafe, None).await?;

    let result = run_tool_effect(
        ctx.clone(),
        &stores.tasks,
        &CancellationToken::new(),
        |_, _collector| async { Err(anyhow::anyhow!("an interrupted call is not run again")) },
        t_plus(20),
    )
    .await;
    assert_interrupted(result)?;

    assert_phases(
        &otel.phases(&ctx.task_id)?,
        &[("lookup", "success"), ("settlement", "success")],
        &ctx,
        &parent,
    );
    Ok(())
}

/// A claim the store refuses is an `error` phase; the fail-closed settle
/// that follows is still timed.
#[tokio::test]
async fn a_refused_claim_is_an_error_phase_before_the_fail_closed_settlement() -> Result<()> {
    let otel = Exporters::install();
    let (parent, traceparent) = otel.task_span();
    let stores = TestStores::new();
    let ctx = tool_child(&stores, "bash", Some(traceparent)).await?;

    let outcome = run_tool_effect(
        ctx.clone(),
        &FailingEffectStart {
            inner: &stores.tasks,
        },
        &CancellationToken::new(),
        |_, _collector| async { Err(anyhow::anyhow!("an unclaimed call is not run")) },
        t_plus(20),
    )
    .await?;
    assert!(matches!(outcome, ToolTaskOutcome::Failed { .. }));

    assert_phases(
        &otel.phases(&ctx.task_id)?,
        &[
            ("lookup", "success"),
            ("claim", "error"),
            ("settlement", "success"),
        ],
        &ctx,
        &parent,
    );
    Ok(())
}

/// A caller running the call under a span of its own gets the phases there,
/// ahead of the stored `traceparent`.
#[tokio::test]
async fn phases_nest_under_the_callers_current_span() -> Result<()> {
    use opentelemetry::context::FutureExt as _;

    let otel = Exporters::install();
    let (_stored, traceparent) = otel.task_span();
    let stores = TestStores::new();
    let ctx = tool_child(&stores, "bash", Some(traceparent)).await?;
    let caller = otel.tracer.tracer("test").start("caller");
    let parent = caller.span_context().clone();

    run_tool_effect(
        ctx.clone(),
        &stores.tasks,
        &CancellationToken::new(),
        |_, _collector| async { Ok(ToolResult::success("ran")) },
        t_plus(20),
    )
    .with_context(opentelemetry::Context::current_with_span(caller))
    .await?;

    let spans = otel.task_spans(&ctx.task_id)?;
    assert_eq!(
        effect_run(&spans)?.parent_span_id,
        parent.span_id(),
        "effect.run nests under the caller's span, beside the phases"
    );

    assert_phases(
        &otel.phases(&ctx.task_id)?,
        &[
            ("lookup", "success"),
            ("claim", "success"),
            ("settlement", "success"),
        ],
        &ctx,
        &parent,
    );
    Ok(())
}

/// The `effect.run` span of `task`'s effect.
fn effect_run(spans: &[SpanData]) -> Result<&SpanData> {
    spans
        .iter()
        .find(|span| span.name == "effect.run")
        .context("the task's effect.run span")
}

/// `effect.run` (from `run_effect`'s signals) nests under the same parent
/// as the phase spans: here the tool child's stored `traceparent`, since
/// the call runs under no span.
#[tokio::test]
async fn effect_run_nests_under_the_same_parent_as_the_phases() -> Result<()> {
    let otel = Exporters::install();
    let (parent, traceparent) = otel.task_span();
    let stores = TestStores::new();
    let ctx = tool_child(&stores, "bash", Some(traceparent)).await?;

    run_tool_effect(
        ctx.clone(),
        &stores.tasks,
        &CancellationToken::new(),
        |_, _collector| async { Ok(ToolResult::success("ran")) },
        t_plus(20),
    )
    .await?;

    let spans = otel.task_spans(&ctx.task_id)?;
    let run = effect_run(&spans)?;
    assert_eq!(run.span_context.trace_id(), parent.trace_id());
    assert_eq!(
        run.parent_span_id,
        parent.span_id(),
        "effect.run nests under the tool task's span, beside the phases"
    );
    assert_phases(
        &otel.phases(&ctx.task_id)?,
        &[
            ("lookup", "success"),
            ("claim", "success"),
            ("settlement", "success"),
        ],
        &ctx,
        &parent,
    );
    Ok(())
}

/// The documented query, `{ span.agent_server.task.id = "<task>" &&
/// (name = "agent_server.tool.phase" || name = "bip.tool.phase") }`, finds
/// every SDK phase of the call, in one trace. (bip's `execution` phase is
/// the `bip.tool.phase` half; bip #985's test covers it.)
#[tokio::test]
async fn one_task_id_query_finds_every_sdk_phase_of_the_call() -> Result<()> {
    let otel = Exporters::install();
    let (parent, traceparent) = otel.task_span();
    let stores = TestStores::new();
    let ctx = tool_child(&stores, "bash", Some(traceparent)).await?;

    run_tool_effect(
        ctx.clone(),
        &stores.tasks,
        &CancellationToken::new(),
        |_, _collector| async { Ok(ToolResult::success("ran")) },
        t_plus(20),
    )
    .await?;

    let spans = otel.task_spans(&ctx.task_id)?;
    let found: Vec<&SpanData> = spans
        .iter()
        .filter(|span| span.name == "agent_server.tool.phase" || span.name == "bip.tool.phase")
        .collect();
    let phases: BTreeSet<Option<String>> = found
        .iter()
        .map(|span| attr(span, "agent_server.tool.phase"))
        .collect();
    assert_eq!(
        phases,
        ["claim", "lookup", "settlement"]
            .into_iter()
            .map(|phase| Some(phase.to_owned()))
            .collect(),
        "the task-id query finds lookup, claim and settlement"
    );
    assert_eq!(found.len(), 3, "each phase once");
    for span in &found {
        assert_eq!(
            attr(span, "gen_ai.tool.call.id").as_deref(),
            Some(ctx.tool_call.id.as_str()),
            "gen_ai.tool.call.id joins the call's phases across emitters"
        );
    }
    // The task's spans, `effect.run` included, are all one trace.
    let traces: std::collections::HashSet<_> = spans
        .iter()
        .map(|span| span.span_context.trace_id())
        .collect();
    assert_eq!(
        traces,
        std::collections::HashSet::from([parent.trace_id()]),
        "one trace per task"
    );
    Ok(())
}
