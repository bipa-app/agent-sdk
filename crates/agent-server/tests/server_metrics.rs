//! Integration tests for `crates/agent-server/src/observability.rs`.
//!
//! Each test installs a fresh in-memory `SdkMeterProvider`, exercises
//! exactly one wire-in point from the Phase 9 B4 card, and asserts
//! the recorded data shape (instrument name + attributes +
//! reasonable bounds on histogram samples).
//!
//! The tests run only with `--features otel` because the
//! observability module itself is gated. When the feature is off
//! the file compiles to an empty translation unit and `cargo test`
//! still passes.

#![cfg(feature = "otel")]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use agent_sdk_foundation::ThreadId;
use anyhow::{Context, Result};
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use time::OffsetDateTime;
use tokio::sync::{Mutex, MutexGuard};

use agent_server::journal::recovery::{FailureReason, RecoveryRecord};
use agent_server::journal::relay::RelayTick;
use agent_server::journal::task::AgentTaskId;
use agent_server::journal::tool_audit::ToolEffectClass;
use agent_server::journal::tool_audit::{
    InMemoryToolAuditEventStore, ToolAuditEvent, ToolAuditEventKind, ToolAuditEventParams,
    ToolAuditEventStore,
};
use agent_server::observability::{ServerMetrics, SubagentStallSignal, ToolStallSignal, attrs};

// ─────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────

/// Tests share the global meter provider and the cached
/// `ServerMetrics` singleton; serialize them so concurrent test
/// runners do not steal each other's exports. Async-aware so
/// `#[tokio::test]` cases can hold the lock across `.await`
/// points cleanly.
static TEST_LOCK: Mutex<()> = Mutex::const_new(());

async fn acquire_test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().await
}

/// Install a fresh meter provider + in-memory exporter and reset
/// the cached `ServerMetrics` singleton so the next
/// `ServerMetrics::global()` call rebuilds against this provider.
fn setup_meter() -> (SdkMeterProvider, InMemoryMetricExporter) {
    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    ServerMetrics::reset_for_testing();
    (provider, exporter)
}

fn collected(exporter: &InMemoryMetricExporter) -> Result<Vec<ResourceMetrics>> {
    exporter
        .get_finished_metrics()
        .context("read collected metrics")
}

fn kv_pairs<'a>(iter: impl Iterator<Item = &'a KeyValue>) -> Vec<(String, String)> {
    iter.map(|kv| (kv.key.as_str().to_string(), format!("{}", kv.value)))
        .collect()
}

fn collect_histogram_attrs(
    snapshots: &[ResourceMetrics],
    metric_name: &str,
) -> Vec<Vec<(String, String)>> {
    let mut out = Vec::new();
    for resource in snapshots {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != metric_name {
                    continue;
                }
                match metric.data() {
                    AggregatedMetrics::F64(MetricData::Histogram(h)) => {
                        for dp in h.data_points() {
                            out.push(kv_pairs(dp.attributes()));
                        }
                    }
                    AggregatedMetrics::U64(MetricData::Histogram(h)) => {
                        for dp in h.data_points() {
                            out.push(kv_pairs(dp.attributes()));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

/// Total number of values recorded across every data point of a
/// histogram. Different from `collect_histogram_attrs(...).len()`,
/// which only counts the *distinct attribute sets* — multiple
/// observations with the same attributes fold into one data point.
fn histogram_total_count(snapshots: &[ResourceMetrics], metric_name: &str) -> u64 {
    let mut total = 0u64;
    for resource in snapshots {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != metric_name {
                    continue;
                }
                match metric.data() {
                    AggregatedMetrics::F64(MetricData::Histogram(h)) => {
                        for dp in h.data_points() {
                            total += dp.count();
                        }
                    }
                    AggregatedMetrics::U64(MetricData::Histogram(h)) => {
                        for dp in h.data_points() {
                            total += dp.count();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    total
}

fn collect_u64_counter(
    snapshots: &[ResourceMetrics],
    metric_name: &str,
) -> Vec<(Vec<(String, String)>, u64)> {
    let mut out = Vec::new();
    for resource in snapshots {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != metric_name {
                    continue;
                }
                if let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() {
                    for dp in sum.data_points() {
                        out.push((kv_pairs(dp.attributes()), dp.value()));
                    }
                }
            }
        }
    }
    out
}

fn collect_i64_up_down_counter(
    snapshots: &[ResourceMetrics],
    metric_name: &str,
) -> Vec<(Vec<(String, String)>, i64)> {
    let mut out = Vec::new();
    for resource in snapshots {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != metric_name {
                    continue;
                }
                if let AggregatedMetrics::I64(MetricData::Sum(sum)) = metric.data() {
                    for dp in sum.data_points() {
                        out.push((kv_pairs(dp.attributes()), dp.value()));
                    }
                }
            }
        }
    }
    out
}

fn collect_u64_gauge(
    snapshots: &[ResourceMetrics],
    metric_name: &str,
) -> Vec<(Vec<(String, String)>, u64)> {
    let mut out = Vec::new();
    for resource in snapshots {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != metric_name {
                    continue;
                }
                if let AggregatedMetrics::U64(MetricData::Gauge(g)) = metric.data() {
                    for dp in g.data_points() {
                        out.push((kv_pairs(dp.attributes()), dp.value()));
                    }
                }
            }
        }
    }
    out
}

fn has_label(set: &[(String, String)], key: &str, value: &str) -> bool {
    set.iter().any(|(k, v)| k == key && v == value)
}

fn matches_all(set: &[(String, String)], expected: &[(&str, &str)]) -> bool {
    expected.iter().all(|(k, v)| has_label(set, k, v))
}

fn flush(provider: &SdkMeterProvider) -> Result<()> {
    provider.force_flush().context("flush meter provider")?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 1. workers_active up/down counter
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn workers_active_balances_started_and_stopped() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    metrics.worker_started();
    metrics.worker_started();
    metrics.worker_started();
    metrics.worker_stopped();

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_i64_up_down_counter(&snapshots, "agent_server.workers.active");
    assert!(!points.is_empty(), "no workers_active points recorded");
    let total: i64 = points.iter().map(|(_, v)| v).sum();
    assert_eq!(total, 2, "started=3, stopped=1, expected balance=2");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 2. tasks.execution.duration always carries kind + outcome
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn task_execution_records_kind_and_outcome() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    metrics.record_task_execution(attrs::KIND_ROOT, attrs::OUTCOME_DONE, 0.42);
    metrics.record_task_execution(attrs::KIND_ROOT, attrs::OUTCOME_SUSPENDED, 0.84);
    metrics.record_task_execution(attrs::KIND_TOOL, attrs::OUTCOME_ERROR, 0.05);
    metrics.record_task_execution(attrs::KIND_SUBAGENT, attrs::OUTCOME_DONE, 1.5);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "agent_server.tasks.execution.duration");
    assert_eq!(points.len(), 4, "expected one point per record call");
    for set in &points {
        // Every sample MUST carry both attributes per the card.
        assert!(
            has_label(set, attrs::KIND, attrs::KIND_ROOT)
                || has_label(set, attrs::KIND, attrs::KIND_TOOL)
                || has_label(set, attrs::KIND, attrs::KIND_SUBAGENT)
        );
        assert!(
            set.iter()
                .any(|(k, v)| k == attrs::OUTCOME && !v.is_empty())
        );
    }

    // tasks.acquired counter increments per kind.
    metrics.record_task_acquired(attrs::KIND_ROOT);
    metrics.record_task_acquired(attrs::KIND_TOOL);
    metrics.record_task_acquired(attrs::KIND_TOOL);
    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.tasks.acquired");
    let root_count: u64 = counts
        .iter()
        .filter(|(s, _)| has_label(s, attrs::KIND, attrs::KIND_ROOT))
        .map(|(_, v)| v)
        .sum();
    let tool_count: u64 = counts
        .iter()
        .filter(|(s, _)| has_label(s, attrs::KIND, attrs::KIND_TOOL))
        .map(|(_, v)| v)
        .sum();
    assert_eq!(root_count, 1);
    assert_eq!(tool_count, 2);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 3. tasks_lease_expired carries the recovery outcome
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn lease_expiry_attributes_each_record_with_reason() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    let id_a = AgentTaskId::from_string("task_a");
    let id_b = AgentTaskId::from_string("task_b");
    let id_c = AgentTaskId::from_string("task_c");
    let records = vec![
        RecoveryRecord::requeued(id_a),
        RecoveryRecord::failed_closed(id_b, FailureReason::RetryBudgetExhausted),
        RecoveryRecord::failed_closed(id_c, FailureReason::UnsafePreparedOperationRecovery),
    ];
    metrics.record_lease_expiry_outcomes(&records);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.tasks.lease_expired");
    let total: u64 = counts.iter().map(|(_, v)| v).sum();
    assert_eq!(total, 3);
    assert!(
        counts
            .iter()
            .any(|(s, _)| has_label(s, attrs::REASON, attrs::REASON_REQUEUED))
    );
    assert!(
        counts
            .iter()
            .any(|(s, _)| has_label(s, attrs::REASON, "retry_budget_exhausted"))
    );
    assert!(counts.iter().any(|(s, _)| has_label(
        s,
        attrs::REASON,
        "unsafe_prepared_operation_recovery"
    )));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 4. journal commit duration tags with event_kind
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn journal_commit_records_event_kind_attribute() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    metrics.record_journal_commit(attrs::COMMIT_KIND_ATOMIC, 0.012);
    metrics.record_journal_commit(attrs::COMMIT_KIND_NON_ATOMIC, 0.045);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "agent_server.journal.commit.duration");
    assert_eq!(points.len(), 2);
    assert!(
        points
            .iter()
            .any(|p| matches_all(p, &[(attrs::EVENT_KIND, attrs::COMMIT_KIND_ATOMIC)]))
    );
    assert!(
        points
            .iter()
            .any(|p| matches_all(p, &[(attrs::EVENT_KIND, attrs::COMMIT_KIND_NON_ATOMIC)]))
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 5. relay_tick fans out the four counters
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn relay_tick_fans_out_to_per_outcome_counter() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    let tick = RelayTick {
        claimed: 5,
        delivered: 3,
        failed: 1,
        expired: 1,
    };
    metrics.record_relay_tick(&tick);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.relay.tick");
    let total: u64 = counts.iter().map(|(_, v)| v).sum();
    assert_eq!(total, 5 + 3 + 1 + 1);

    for outcome in [
        attrs::RELAY_CLAIMED,
        attrs::RELAY_DELIVERED,
        attrs::RELAY_FAILED,
        attrs::RELAY_EXPIRED,
    ] {
        assert!(
            counts
                .iter()
                .any(|(s, _)| has_label(s, attrs::RELAY_OUTCOME, outcome)),
            "missing relay outcome {outcome}",
        );
    }
    Ok(())
}

#[tokio::test]
async fn relay_tick_skips_zero_outcomes() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    let tick = RelayTick {
        claimed: 0,
        delivered: 0,
        failed: 0,
        expired: 0,
    };
    metrics.record_relay_tick(&tick);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.relay.tick");
    assert!(
        counts.is_empty(),
        "no points expected for an all-zero tick, got {counts:?}",
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 6. wakeup fallback sweep counter
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn wakeup_fallback_sweep_increments_per_pulse() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    metrics.record_fallback_sweep();
    metrics.record_fallback_sweep();
    metrics.record_fallback_sweep();

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.wakeup.fallback_sweep");
    let total: u64 = counts.iter().map(|(_, v)| v).sum();
    assert_eq!(total, 3);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 7. thread events watch lag
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn thread_events_watch_lag_records_milliseconds() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    metrics.record_thread_events_watch_lag(2.5);
    metrics.record_thread_events_watch_lag(120.0);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    // Two samples land into the same `()`-attributed data point, so
    // assert on the total recorded count rather than the number of
    // distinct attribute sets.
    let total = histogram_total_count(&snapshots, "agent_server.thread_events_watch.lag_ms");
    assert_eq!(total, 2);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 8. tool audit outcome counter — driven through the in-memory store
// ─────────────────────────────────────────────────────────────────────

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(1_700_000_000)
}

fn audit_event(kind: ToolAuditEventKind, effect_class: ToolEffectClass) -> ToolAuditEvent {
    ToolAuditEvent::new(ToolAuditEventParams {
        operation_id: "op_1".into(),
        task_id: AgentTaskId::from_string("task_audit"),
        parent_task_id: AgentTaskId::from_string("task_parent"),
        thread_id: ThreadId::from_string("t-audit"),
        tool_call_id: "call_1".into(),
        tool_name: "noop".into(),
        effect_class,
        kind,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-5-20250929".into(),
        input: None,
        output: None,
        error: None,
        now: t0(),
    })
}

#[tokio::test]
async fn tool_audit_outcome_counter_records_kind_and_effect_class() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    // Force-init metrics so the singleton binds to this provider
    // before the first audit event lands.
    let _ = ServerMetrics::global();

    let store = InMemoryToolAuditEventStore::new();
    store
        .record_event(&audit_event(
            ToolAuditEventKind::ExecutionStarted,
            ToolEffectClass::SideEffecting,
        ))
        .await?;
    store
        .record_event(&audit_event(
            ToolAuditEventKind::Completed,
            ToolEffectClass::SideEffecting,
        ))
        .await?;
    store
        .record_event(&audit_event(
            ToolAuditEventKind::FailClosed {
                reason: "intent_persist".into(),
            },
            ToolEffectClass::ReplaySafe,
        ))
        .await?;

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.tool_audit.outcome");
    let total: u64 = counts.iter().map(|(_, v)| v).sum();
    assert_eq!(total, 3);

    assert!(counts.iter().any(|(s, _)| matches_all(
        s,
        &[
            (attrs::AUDIT_OUTCOME, "execution_started"),
            (attrs::EFFECT_CLASS, "side_effecting"),
        ],
    )));
    assert!(counts.iter().any(|(s, _)| matches_all(
        s,
        &[
            (attrs::AUDIT_OUTCOME, "completed"),
            (attrs::EFFECT_CLASS, "side_effecting"),
        ],
    )));
    assert!(counts.iter().any(|(s, _)| matches_all(
        s,
        &[
            (attrs::AUDIT_OUTCOME, "fail_closed"),
            (attrs::EFFECT_CLASS, "replay_safe"),
        ],
    )));
    Ok(())
}

// ─────────────────────────────────────────���───────────────────────────
// 9. outbox depth gauge — caller-supplied closure publishes the value
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn outbox_depth_gauge_publishes_callback_value() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    let depth = Arc::new(AtomicU64::new(7));
    let depth_for_callback = Arc::clone(&depth);
    let _gauge =
        metrics.register_outbox_depth_callback(move || depth_for_callback.load(Ordering::Relaxed));

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_u64_gauge(&snapshots, "agent_server.journal.outbox_depth");
    assert!(!points.is_empty(), "no gauge points published");
    let last = points.last().context("at least one gauge point")?.1;
    assert_eq!(last, 7);

    // Bump the depth and re-export — the callback should pull the
    // latest value rather than reusing a cached snapshot.
    depth.store(42, Ordering::Relaxed);
    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_u64_gauge(&snapshots, "agent_server.journal.outbox_depth");
    assert!(points.iter().any(|(_, v)| *v == 42));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// 10. Spot-check that durations are recorded in the right unit
// ──────────────────────────────────────────────────────────────────��──

#[tokio::test]
async fn duration_helpers_record_in_seconds_and_milliseconds() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();
    let started = std::time::Instant::now();
    std::thread::sleep(Duration::from_millis(2));
    let elapsed = started.elapsed();
    metrics.record_task_execution(attrs::KIND_ROOT, attrs::OUTCOME_DONE, elapsed.as_secs_f64());
    metrics.record_thread_events_watch_lag(elapsed.as_secs_f64() * 1_000.0);
    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    assert_ne!(
        collect_histogram_attrs(&snapshots, "agent_server.tasks.execution.duration"),
        [] as [Vec<(String, String)>; 0]
    );
    assert_ne!(
        collect_histogram_attrs(&snapshots, "agent_server.thread_events_watch.lag_ms"),
        [] as [Vec<(String, String)>; 0]
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// transitions and documents, recorded by the store's commit
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_commit_records_its_outcome_and_its_documents() -> Result<()> {
    use agent_server::journal::store::InMemoryAgentTaskStore;
    use agent_server::journal::{
        DocExpect, DocKind, DocScope, DocumentKey, DocumentWrite, DurableStore, Transition, Write,
    };

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let store = InMemoryAgentTaskStore::new();
    let key = DocumentKey::new(DocScope::Host, DocKind::new("test.note")?, "k");
    let put = |expect| Transition {
        guards: Vec::new(),
        writes: vec![Write::Document(DocumentWrite::put(
            key.clone(),
            1,
            expect,
            serde_json::json!(1),
        ))],
    };
    let now = OffsetDateTime::UNIX_EPOCH;
    store
        .commit(put(DocExpect::Absent), now)
        .await
        .map_err(anyhow::Error::from)?;
    let rejected = store.commit(put(DocExpect::Absent), now).await;
    assert!(rejected.is_err(), "a second Absent put is rejected");

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let points = collect_histogram_attrs(&snapshots, "agent_server.journal.transition.duration");
    for outcome in [attrs::OUTCOME_COMMITTED, attrs::OUTCOME_EXPECTATION] {
        assert!(
            points.iter().any(|p| matches_all(
                p,
                &[
                    (attrs::BACKEND, attrs::BACKEND_MEMORY),
                    (attrs::OUTCOME, outcome)
                ]
            )),
            "a {outcome} transition is recorded: {points:?}"
        );
    }
    let written = collect_u64_counter(&snapshots, "agent_server.journal.documents.written");
    assert_eq!(written.len(), 1, "{written:?}");
    let (labels, count) = &written[0];
    assert_eq!(*count, 1);
    assert!(matches_all(
        labels,
        &[
            (attrs::BACKEND, attrs::BACKEND_MEMORY),
            (attrs::SCOPE, "host"),
            (attrs::OP, attrs::OP_PUT),
        ]
    ));
    Ok(())
}

#[tokio::test]
async fn a_transition_extension_records_its_writes_and_refusals() -> Result<()> {
    use agent_server::journal::store::InMemoryAgentTaskStore;
    use agent_server::journal::{
        DocExpect, DocKind, DocScope, DocumentKey, DocumentWrite, DurableStore, Transition,
        TransitionExtension, TransitionView, Write,
    };

    struct AddsADocument {
        refuse: bool,
    }

    impl TransitionExtension for AddsADocument {
        fn extend(&self, _view: TransitionView<'_>) -> Vec<Write> {
            let Ok(kind) = DocKind::new("test.extension") else {
                return Vec::new();
            };
            let mut writes = vec![Write::Document(DocumentWrite::put(
                DocumentKey::new(DocScope::Host, kind, "added"),
                1,
                DocExpect::Any,
                serde_json::json!(1),
            ))];
            if self.refuse {
                writes.push(Write::AppendCompaction {
                    thread: ThreadId::from_string("t"),
                    messages: Vec::new(),
                    source_message_count: 0,
                    retained_message_count: 0,
                });
            }
            writes
        }
    }

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let now = OffsetDateTime::UNIX_EPOCH;
    for refuse in [false, true] {
        let store = InMemoryAgentTaskStore::new()
            .with_transition_extension(Arc::new(AddsADocument { refuse }));
        let result = store.commit(Transition::default(), now).await;
        assert_eq!(result.is_err(), refuse);
    }

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let added = collect_u64_counter(&snapshots, "agent_server.journal.extension.writes");
    assert_eq!(added.len(), 1, "{added:?}");
    assert!(matches_all(
        &added[0].0,
        &[(attrs::EXTENSION_POINT, attrs::EXTENSION_POINT_COMMIT)]
    ));
    assert_eq!(added[0].1, 1);
    let refused = collect_u64_counter(&snapshots, "agent_server.journal.extension.refused");
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert!(matches_all(
        &refused[0].0,
        &[
            (attrs::EXTENSION_POINT, attrs::EXTENSION_POINT_COMMIT),
            (attrs::WRITE_KIND, "AppendCompaction"),
        ]
    ));
    assert_eq!(refused[0].1, 1);
    Ok(())
}

/// A root on `thread`, parked on one tool child.
async fn parked_root(
    store: &agent_server::journal::store::InMemoryAgentTaskStore,
    thread: &ThreadId,
    now: OffsetDateTime,
) -> Result<agent_server::journal::task::AgentTask> {
    use agent_sdk_foundation::{AgentContinuation, AgentState, ContinuationEnvelope, TokenUsage};
    use agent_server::journal::task::{
        AgentTask, ChildSpawnSpec, LeaseId, SuspensionPayload, WorkerId,
    };
    use agent_server::journal::{AgentTaskStore, SpawnChildrenExt};

    let root = AgentTask::new_root_turn(thread.clone(), now, 3);
    let _ = store.submit_root_turn(root.clone()).await?;
    let (worker, lease) = (WorkerId::new(), LeaseId::new());
    let _ = store
        .try_acquire_task(
            &root.id,
            worker.clone(),
            lease.clone(),
            now + Duration::from_secs(60),
            now,
        )
        .await?;
    let payload = SuspensionPayload {
        continuation: ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread.clone()),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        }),
        suspended_messages: Vec::new(),
        child_join_policy: agent_server::ChildJoinPolicy::default(),
    };
    let _ = store
        .spawn_tool_children(
            &root.id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(3)],
            payload,
            None,
            Vec::new(),
            now,
        )
        .await?;
    Ok(root)
}

#[tokio::test]
async fn a_steering_wake_records_whether_it_woke_the_parent() -> Result<()> {
    use agent_sdk_foundation::llm::ContentBlock;
    use agent_server::journal::store::InMemoryAgentTaskStore;
    use agent_server::journal::{DurableStore, Transition};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let store = InMemoryAgentTaskStore::new();
    let now = OffsetDateTime::UNIX_EPOCH;
    let thread = ThreadId::from_string("steer");
    let root = parked_root(&store, &thread, now).await?;
    let note = || {
        vec![ContentBlock::Text {
            text: "status?".into(),
        }]
    };
    for expected_woken in [true, false] {
        let committed = store
            .commit(
                Transition::enqueue_steering_resume(root.id.clone(), note()),
                now,
            )
            .await
            .map_err(anyhow::Error::from)?;
        assert_eq!(
            matches!(committed.steering_resumed.as_slice(), [Some(_)]),
            expected_woken,
            "the parked parent wakes, then a second wake declines"
        );
    }

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let resumes = collect_u64_counter(&snapshots, "agent_server.journal.steering_resumes");
    assert_eq!(resumes.len(), 2, "{resumes:?}");
    for outcome in [attrs::OUTCOME_WOKEN, attrs::OUTCOME_DECLINED] {
        assert!(
            resumes.iter().any(|(labels, count)| *count == 1
                && matches_all(
                    labels,
                    &[
                        (attrs::BACKEND, attrs::BACKEND_MEMORY),
                        (attrs::OUTCOME, outcome)
                    ]
                )),
            "one {outcome} wake is counted: {resumes:?}"
        );
    }
    Ok(())
}

/// A queued steering wake counts each read of its queue by outcome:
/// nothing queued, taken, then the parent no longer parked.
#[tokio::test]
async fn a_queued_steering_wake_records_each_read_of_its_queue() -> Result<()> {
    use agent_sdk_foundation::llm::ContentBlock;
    use agent_server::journal::store::InMemoryAgentTaskStore;
    use agent_server::journal::task::AgentTask;
    use agent_server::journal::{
        DocExpect, DocKind, DocScope, Document, DocumentKey, DocumentWrite, DurableStore,
        QueuedSteering, SteeringInput, Transition, TransitionExtension, TransitionView, Write,
    };

    struct TakesTheQueue;

    impl TransitionExtension for TakesTheQueue {
        fn extend(&self, _view: TransitionView<'_>) -> Vec<Write> {
            Vec::new()
        }

        fn take_queued_steering(
            &self,
            _parent: &AgentTask,
            queued: &Document,
        ) -> Option<QueuedSteering> {
            Some(QueuedSteering {
                steering: vec![ContentBlock::Text {
                    text: queued.value.to_string(),
                }],
                schema_version: queued.schema_version,
                value: serde_json::json!(null),
            })
        }
    }

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let store = InMemoryAgentTaskStore::new().with_transition_extension(Arc::new(TakesTheQueue));
    let now = OffsetDateTime::UNIX_EPOCH;
    let thread = ThreadId::from_string("queued");
    let root = parked_root(&store, &thread, now).await?;
    let kind = DocKind::new("test.steering")?;
    let wake = || Transition {
        guards: Vec::new(),
        writes: vec![Write::EnqueueSteeringResume {
            parent: root.id.clone(),
            steering: SteeringInput::Queued {
                kind: kind.clone(),
                key: String::new(),
            },
        }],
    };
    let _ = store
        .commit(wake(), now)
        .await
        .map_err(anyhow::Error::from)?;
    let _ = store
        .commit(
            Transition {
                guards: Vec::new(),
                writes: vec![Write::Document(DocumentWrite::put(
                    DocumentKey::new(DocScope::Thread(thread.clone()), kind.clone(), ""),
                    1,
                    DocExpect::Absent,
                    serde_json::json!(["status?"]),
                ))],
            },
            now,
        )
        .await
        .map_err(anyhow::Error::from)?;
    for _ in 0..2 {
        let _ = store
            .commit(wake(), now)
            .await
            .map_err(anyhow::Error::from)?;
    }

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let takes = collect_u64_counter(&snapshots, "agent_server.journal.steering_queue.takes");
    assert_eq!(takes.len(), 3, "{takes:?}");
    for outcome in [
        attrs::OUTCOME_EMPTY,
        attrs::OUTCOME_TAKEN,
        attrs::OUTCOME_NOT_PARKED,
    ] {
        assert!(
            takes.iter().any(|(labels, count)| *count == 1
                && matches_all(
                    labels,
                    &[
                        (attrs::BACKEND, attrs::BACKEND_MEMORY),
                        (attrs::OUTCOME, outcome)
                    ]
                )),
            "one {outcome} read is counted: {takes:?}"
        );
    }
    Ok(())
}

/// A store error marked [`LockConflict`](agent_server::journal::LockConflict)
/// is recorded as `outcome=lock_conflict`; any other store error stays
/// `outcome=store`.
#[tokio::test]
async fn a_lock_conflict_the_store_gave_up_on_is_its_own_outcome() -> Result<()> {
    use agent_server::journal::{LockConflict, Rejected};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();
    let gave_up = Err(Rejected::Store(
        anyhow::anyhow!("deadlock detected").context(LockConflict { retries: 3 }),
    ));
    let other = Err(Rejected::Store(anyhow::anyhow!("connection reset")));
    // Different backends, so the test sees which error got which outcome.
    metrics.record_transition(attrs::BACKEND_POSTGRES, 0, &gave_up, 0.03);
    metrics.record_transition(attrs::BACKEND_SQLITE, 0, &other, 0.01);

    flush(&provider)?;
    let points = collect_histogram_attrs(
        &collected(&exporter)?,
        "agent_server.journal.transition.duration",
    );
    let recorded = |backend: &str, outcome: &str| {
        points
            .iter()
            .any(|p| matches_all(p, &[(attrs::BACKEND, backend), (attrs::OUTCOME, outcome)]))
    };
    assert!(
        recorded(attrs::BACKEND_POSTGRES, attrs::OUTCOME_LOCK_CONFLICT),
        "the marked error is a lock conflict: {points:?}"
    );
    assert!(
        recorded(attrs::BACKEND_SQLITE, attrs::OUTCOME_STORE),
        "an unmarked store error stays a store failure: {points:?}"
    );
    assert!(
        !recorded(attrs::BACKEND_POSTGRES, attrs::OUTCOME_STORE)
            && !recorded(attrs::BACKEND_SQLITE, attrs::OUTCOME_LOCK_CONFLICT),
        "{points:?}"
    );
    Ok(())
}

fn tool_call_effects(counts: &[(Vec<(String, String)>, u64)], transition: &str) -> u64 {
    counts
        .iter()
        .filter(|(set, _)| {
            matches_all(
                set,
                &[(attrs::KIND, "tool_call"), (attrs::TRANSITION, transition)],
            )
        })
        .map(|(_, value)| value)
        .sum()
}

#[tokio::test]
async fn effect_signals_count_each_transition_by_kind() -> Result<()> {
    use agent_sdk_foundation::ToolReplay;
    use agent_server::journal::effect::{
        EffectIntent, EffectKey, EffectRun, EffectState, EffectSubject, run_effect,
    };
    use agent_server::journal::store::{AgentTaskStore, InMemoryAgentTaskStore};
    use agent_server::journal::task::{AgentTask, LeaseId, TaskStatus, WorkerId};
    use agent_server::journal::{DurableStore, Guard, Settle, Transition};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let _ = ServerMetrics::global();

    let store = InMemoryAgentTaskStore::new();
    let task = AgentTask::new_root_turn(ThreadId::from_string("effect-metrics"), t0(), 3);
    store.submit_root_turn(task.clone()).await?;
    let (worker, lease) = (WorkerId::new(), LeaseId::new());
    store
        .try_acquire_task(
            &task.id,
            worker.clone(),
            lease.clone(),
            t0() + time::Duration::hours(1),
            t0(),
        )
        .await?
        .context("acquire the owner")?;
    let intent = |call: &str| EffectIntent {
        key: EffectKey::tool(&task.id, call),
        replay: ToolReplay::Unsafe,
        owner: task.id.clone(),
        thread: task.thread_id.clone(),
        subject: EffectSubject::Tool {
            call_id: call.into(),
            name: "bash".into(),
            input: serde_json::json!({}),
        },
        attempt: 1,
    };
    let leased = || Transition {
        guards: vec![Guard::TaskIs {
            task: task.id.clone(),
            status: TaskStatus::Running,
            lease: Some((worker.clone(), lease.clone())),
        }],
        writes: Vec::new(),
    };
    let run = |call: &'static str| {
        run_effect(
            &store,
            intent(call),
            leased(),
            || async {},
            |()| anyhow::Ok((Transition::default(), EffectState::Completed)),
            t0(),
        )
    };

    assert!(matches!(run("call_1").await?, EffectRun::Ran { .. }));
    assert!(matches!(run("call_1").await?, EffectRun::Refused(_)));
    let mut start_only = leased();
    start_only.guards.push(Guard::EffectFree {
        key: EffectKey::tool(&task.id, "call_2"),
    });
    start_only.writes.push(agent_server::journal::Write::Effect(
        agent_server::journal::EffectWrite::Start(intent("call_2")),
    ));
    store.commit(start_only, t0()).await?;
    store
        .commit(
            Transition::settle_only(
                task.id.clone(),
                worker,
                lease,
                Settle::Fail { error: "x".into() },
            ),
            t0(),
        )
        .await?;

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.effects");
    for transition in ["started", "finished", "refused", "abandoned"] {
        assert_eq!(
            tool_call_effects(&counts, transition),
            1,
            "transition {transition}: {counts:?}"
        );
    }
    let durations = collect_histogram_attrs(&snapshots, "agent_server.effect.duration");
    assert!(
        durations.iter().any(|set| matches_all(
            set,
            &[(attrs::KIND, "tool_call"), (attrs::OUTCOME, "completed")]
        )),
        "{durations:?}"
    );
    assert_eq!(
        histogram_total_count(&snapshots, "agent_server.effect.duration"),
        1
    );
    Ok(())
}

/// An in-memory task store wired to commit completed turns, as a host
/// wires it, with the attempt store it closes attempts in.
fn committing_store() -> (
    agent_server::journal::store::InMemoryAgentTaskStore,
    agent_server::journal::turn_attempt_store::InMemoryTurnAttemptStore,
) {
    use agent_server::journal::checkpoint_store::InMemoryCheckpointStore;
    use agent_server::journal::event_repository::InMemoryEventRepository;
    use agent_server::journal::message_store::InMemoryMessageProjectionStore;
    use agent_server::journal::outbox::InMemoryOutboxStore;
    use agent_server::journal::store::{
        CancellationMarkerSink, CompletedTurnSink, InMemoryAgentTaskStore,
    };
    use agent_server::journal::thread_store::InMemoryThreadStore;
    use agent_server::journal::turn_attempt_store::InMemoryTurnAttemptStore;

    let attempts = InMemoryTurnAttemptStore::new();
    let store = InMemoryAgentTaskStore::new()
        .with_cancellation_markers(CancellationMarkerSink {
            event_repo: Arc::new(InMemoryEventRepository::new()),
            outbox_store: Arc::new(InMemoryOutboxStore::new()),
            thread_store: Arc::new(InMemoryThreadStore::new()),
        })
        .with_message_store(Arc::new(InMemoryMessageProjectionStore::new()))
        .with_completed_turns(CompletedTurnSink {
            attempt_store: Arc::new(attempts.clone()),
            checkpoint_store: Arc::new(InMemoryCheckpointStore::new()),
        });
    (store, attempts)
}

fn empty_close() -> agent_server::journal::turn_attempt::CloseAttemptParams {
    use agent_server::journal::turn_attempt::{CloseAttemptParams, TurnAttemptOutcome};

    CloseAttemptParams {
        response_blob: serde_json::json!({}),
        response_id: None,
        response_model: None,
        stop_reason: None,
        outcome: TurnAttemptOutcome::Success,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
        route_provider: None,
        thinking_mode: None,
        thinking_budget_tokens: None,
        thinking_effort: None,
    }
}

/// Every completed-turn commit is counted with its bounded outcome, the
/// refusals included: a worker that lost its lease discards a finished turn,
/// so its rate matters.
#[tokio::test]
async fn completed_turn_commits_count_each_outcome() -> Result<()> {
    use agent_sdk_foundation::audit::AuditProvenance;
    use agent_sdk_foundation::{TokenUsage, llm};
    use agent_server::journal::checkpoint::CheckpointKind;
    use agent_server::journal::commit::{
        CommitOwnerGuard, CompletedTurnCommit, commit_completed_turn,
    };
    use agent_server::journal::store::AgentTaskStore;
    use agent_server::journal::task::{AgentTask, LeaseId, WorkerId};
    use agent_server::journal::turn_attempt::OpenAttemptParams;
    use agent_server::journal::turn_attempt_store::TurnAttemptStore;
    use agent_server::journal::{Settle, Write};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let _ = ServerMetrics::global();

    let (store, attempts) = committing_store();
    let thread = ThreadId::from_string("completed-turn-metrics");
    let root = AgentTask::new_root_turn(thread.clone(), t0(), 3);
    store.submit_root_turn(root.clone()).await?;
    let (worker, lease) = (WorkerId::new(), LeaseId::new());
    store
        .try_acquire_task(
            &root.id,
            worker.clone(),
            lease.clone(),
            t0() + time::Duration::hours(1),
            t0(),
        )
        .await?
        .context("acquire the root")?;
    let attempt = attempts
        .open_attempt(OpenAttemptParams {
            task_id: root.id.clone(),
            attempt_number: 1,
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-5-5"),
            request_blob: serde_json::json!({}),
            now: t0(),
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await?;
    let commit = |expected_turn: u32, lease: LeaseId| CompletedTurnCommit {
        extra_writes: vec![Write::SettleTask {
            task: root.id.clone(),
            outcome: Settle::CompleteWithoutResult,
        }],
        delivered_injection_ids: Vec::new(),
        checkpoint_kind: CheckpointKind::FullTurn,
        thread_id: thread.clone(),
        task_id: root.id.clone(),
        expected_turn,
        turn_attempt_id: attempt.id.clone(),
        close_attempt_params: empty_close(),
        messages: vec![llm::Message::assistant("done")],
        turn_usage: TokenUsage::default(),
        agent_state_snapshot: serde_json::json!({}),
        events: Vec::new(),
        outbox_max_attempts: 3,
        owner_guard: Some(CommitOwnerGuard {
            worker_id: worker.clone(),
            lease_id: lease,
        }),
        now: t0(),
    };

    assert!(
        commit_completed_turn(commit(1, LeaseId::new()), &store)
            .await
            .is_err()
    );
    assert!(
        commit_completed_turn(commit(2, lease.clone()), &store)
            .await
            .is_err()
    );
    commit_completed_turn(commit(1, lease), &store).await?;

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.journal.completed_turn.commits");
    for outcome in ["lost_ownership", "stale_turn", "ok"] {
        let seen: u64 = counts
            .iter()
            .filter(|(set, _)| has_label(set, attrs::OUTCOME, outcome))
            .map(|(_, value)| value)
            .sum();
        assert_eq!(seen, 1, "outcome {outcome}: {counts:?}");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Read-time session view: the fold and stream recorders the view RPCs call
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn view_signals_record_the_fold_and_stream_steps() -> Result<()> {
    use agent_server::journal::view::signals;

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    signals::folded(1_500, 0.004);
    signals::stream(signals::OPENED);
    signals::stream(signals::RESNAPSHOT);
    signals::stream(signals::NOT_FOUND);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    assert_eq!(
        histogram_total_count(&snapshots, "agent_server.view.fold.duration"),
        1
    );
    assert_eq!(
        histogram_total_count(&snapshots, "agent_server.view.folded_events"),
        1
    );
    let streams = collect_u64_counter(&snapshots, "agent_server.view.stream");
    for event in ["opened", "resnapshot", "not_found"] {
        assert!(
            streams
                .iter()
                .any(|(set, _)| has_label(set, attrs::VIEW_STREAM_EVENT, event)),
            "{event}: {streams:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_prompt_step_is_counted_by_owner_step_and_outcome() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = agent_server::observability::ServerMetrics::global();
    metrics.record_prompt(
        "host",
        agent_server::journal::PromptStep::Answer,
        attrs::OUTCOME_COMMITTED,
    );
    metrics.record_prompt(
        "host",
        agent_server::journal::PromptStep::Answer,
        attrs::OUTCOME_COMMITTED,
    );
    metrics.record_prompt(
        "task",
        agent_server::journal::PromptStep::Withdraw,
        attrs::OUTCOME_EXPECTATION,
    );

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counted = collect_u64_counter(&snapshots, "agent_server.prompts");
    let count = |owner: &str, step: &str, outcome: &str| {
        counted
            .iter()
            .filter(|(labels, _)| {
                matches_all(
                    labels,
                    &[
                        (attrs::PROMPT_OWNER, owner),
                        (attrs::PROMPT_STEP, step),
                        (attrs::OUTCOME, outcome),
                    ],
                )
            })
            .map(|(_, count)| *count)
            .sum::<u64>()
    };
    assert_eq!(
        count("host", "answer", attrs::OUTCOME_COMMITTED),
        2,
        "{counted:?}"
    );
    assert_eq!(
        count("task", "withdraw", attrs::OUTCOME_EXPECTATION),
        1,
        "{counted:?}"
    );
    Ok(())
}

#[tokio::test]
async fn tool_stall_signals_count_by_outcome_without_the_tool_name() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    let stall = |tool_name: &'static str, task_id: &'static str| ToolStallSignal {
        tool_name,
        task_id,
        traceparent: None,
        budget_ms: 900_000,
        silent_for_ms: 901_000,
    };
    metrics.record_tool_stall(attrs::STALL_TRIPPED, &stall("mcp__any__tool", "task-1"));
    metrics.record_tool_stall(attrs::STALL_TRIPPED, &stall("bash", "task-2"));
    metrics.record_tool_stall(attrs::STALL_FALLBACK_SETTLED, &stall("bash", "task-2"));
    metrics.record_tool_stall_unbounded(&stall("watch_logs", "task-3"));

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let stalls = collect_u64_counter(&snapshots, "agent_server.tool.stalls");
    let by_outcome = |outcome: &str| -> u64 {
        stalls
            .iter()
            .filter(|(set, _)| matches_all(set, &[(attrs::OUTCOME, outcome)]))
            .map(|(_, count)| count)
            .sum()
    };
    assert_eq!(by_outcome(attrs::STALL_TRIPPED), 2);
    assert_eq!(by_outcome(attrs::STALL_FALLBACK_SETTLED), 1);
    assert!(
        stalls.iter().all(|(set, _)| set.len() == 1),
        "the counter carries only its bounded outcome: {stalls:?}"
    );
    let unbounded = collect_u64_counter(&snapshots, "agent_server.tool.stall_budget.unbounded");
    assert_eq!(unbounded.iter().map(|(_, count)| count).sum::<u64>(), 1);
    assert!(unbounded.iter().all(|(set, _)| set.is_empty()));
    Ok(())
}

#[tokio::test]
async fn progress_events_and_refusals_carry_only_their_bounded_label() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    metrics.record_progress_event(attrs::PROGRESS_SUBAGENT);
    metrics.record_progress_event(attrs::PROGRESS_SUBAGENT);
    metrics.record_progress_event(attrs::PROGRESS_ROOT_STALL_NOTICE);
    metrics.record_progress_event(attrs::PROGRESS_USAGE);
    metrics.record_progress_refused(attrs::PROGRESS_LEASE_LOST);
    metrics.record_progress_refused(attrs::PROGRESS_PURGED);
    metrics.record_progress_refused(attrs::PROGRESS_STORE);
    metrics.record_progress_refused(attrs::PROGRESS_LEASE_LOST);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let sum = |name: &str, key: &str, value: &str| -> u64 {
        collect_u64_counter(&snapshots, name)
            .iter()
            .filter(|(set, _)| matches_all(set, &[(key, value)]))
            .map(|(_, count)| count)
            .sum()
    };
    let events = "agent_server.progress.events";
    assert_eq!(sum(events, attrs::KIND, "subagent_progress"), 2);
    assert_eq!(sum(events, attrs::KIND, "root_stall_notice"), 1);
    assert_eq!(sum(events, attrs::KIND, "usage_progress"), 1);
    let refused = "agent_server.progress.refused";
    assert_eq!(sum(refused, attrs::REASON, "lease_lost"), 2);
    assert_eq!(sum(refused, attrs::REASON, "purged"), 1);
    assert_eq!(sum(refused, attrs::REASON, "store"), 1);
    for name in [events, refused] {
        let points = collect_u64_counter(&snapshots, name);
        assert!(
            points.iter().all(|(set, _)| set.len() == 1),
            "{name} carries only its one bounded label: {points:?}"
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// fence.refused counts each refused writer by thread status
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_fence_refusal_is_counted_by_writer_and_status() -> Result<()> {
    use agent_server::journal::Fence;
    use agent_server::journal::thread::{
        FencedWriter, PurgeScope, PurgeSeed, Thread, record_fence_refusal,
    };

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let thread_id = ThreadId::from_string("fenced-thread");
    let now = OffsetDateTime::UNIX_EPOCH;
    let deleting = Thread::new(thread_id.clone(), now).begin_purge(PurgeSeed {
        root_thread_id: thread_id.clone(),
        scope: PurgeScope::Thread,
        started_at: now,
    });

    for writer in [FencedWriter::MessageHead, FencedWriter::PauseOnQuestion] {
        let refused = Fence::APPEND.check(&thread_id, Some(&deleting));
        assert!(record_fence_refusal(writer, refused).is_err());
    }
    assert!(
        record_fence_refusal(
            FencedWriter::ForkSource,
            Fence::FORK_SOURCE.check(&thread_id, None)
        )
        .is_ok()
    );
    assert!(
        record_fence_refusal(
            FencedWriter::RequeueOwnedTask,
            Err(anyhow::anyhow!("not a fence"))
        )
        .is_err()
    );

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.fence.refused");
    let total: u64 = counts.iter().map(|(_, v)| v).sum();
    assert_eq!(total, 2, "only the fence's own refusals count: {counts:?}");
    for writer in ["message_head", "pause_on_question"] {
        assert!(
            counts
                .iter()
                .any(|(s, _)| has_label(s, attrs::WRITER, writer)
                    && has_label(s, attrs::THREAD_STATUS, "deleting")),
            "{writer}: {counts:?}"
        );
    }
    Ok(())
}

/// A subagent invocation's run is counted by attempt on
/// `agent_server.subagent.invocation.attempts`, and a retry (attempt 2,
/// after the first lease expired) also emits the span
/// `agent_server.subagent.invocation.retry` with the task's id, kind,
/// attempt and budget.
#[tokio::test]
async fn an_invocation_retry_is_counted_by_attempt_and_traced() -> Result<()> {
    use agent_server::journal::store::AgentTaskStore;
    use agent_server::journal::task::{LeaseId, WorkerId};
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let spans = InMemorySpanExporter::default();
    let tracer_provider = SdkTracerProvider::builder()
        .with_simple_exporter(spans.clone())
        .build();
    opentelemetry::global::set_tracer_provider(tracer_provider.clone());
    let _ = tracer_provider.tracer("warm-up");
    let _ = ServerMetrics::global();

    let stores = invocation_fixture::Stores::new();
    let invocation = stores.runnable_invocation("invocation-retry").await?;
    stores
        .tasks
        .try_acquire_task(
            &invocation.id,
            WorkerId::new(),
            LeaseId::new(),
            t0() + time::Duration::seconds(5),
            t0(),
        )
        .await?
        .context("attempt 1 acquires")?;
    stores
        .tasks
        .release_expired_leases(
            t0() + time::Duration::seconds(6),
            agent_server::journal::LeaseRecoveryScope::Expired,
        )
        .await?;
    let outcome = stores
        .run_invocation(&invocation.id, t0() + time::Duration::seconds(10))
        .await?;
    assert_eq!(outcome.invocation_task.attempt, 2);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let attempts = collect_u64_counter(&snapshots, "agent_server.subagent.invocation.attempts");
    assert_eq!(attempts.len(), 1, "{attempts:?}");
    assert!(matches_all(&attempts[0].0, &[(attrs::ATTEMPT, "2")]));
    assert_eq!(attempts[0].1, 1);

    tracer_provider
        .force_flush()
        .context("flush tracer provider")?;
    let finished = spans.get_finished_spans().context("read spans")?;
    let retry = finished
        .iter()
        .find(|span| span.name == "agent_server.subagent.invocation.retry")
        .with_context(|| format!("no retry span in {finished:?}"))?;
    let attributes: Vec<(String, String)> = retry
        .attributes
        .iter()
        .map(|kv| (kv.key.as_str().to_owned(), kv.value.to_string()))
        .collect();
    let invocation_id = invocation.id.as_str().to_owned();
    assert!(
        matches_all(
            &attributes,
            &[
                (attrs::SPAN_TASK_ID, invocation_id.as_str()),
                (attrs::SPAN_TASK_KIND, "subagent"),
                (attrs::SPAN_TASK_ATTEMPT, "2"),
                (attrs::SPAN_TASK_MAX_ATTEMPTS, "3"),
            ]
        ),
        "{attributes:?}"
    );
    Ok(())
}

/// In-memory stores and a runnable subagent invocation: its parent waits
/// on one pending `subagent` call, and its child root was cancelled.
mod invocation_fixture {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use agent_sdk_foundation::audit::AuditProvenance;
    use agent_sdk_foundation::{
        AgentContinuation, AgentState, ContinuationEnvelope, PendingToolCallInfo, ThreadId,
        TokenUsage, ToolConcurrency, ToolReplay, ToolTier,
    };
    use agent_server::journal::SpawnChildrenExt;
    use agent_server::journal::event_repository::InMemoryEventRepository;
    use agent_server::journal::message_store::InMemoryMessageProjectionStore;
    use agent_server::journal::outbox::InMemoryOutboxStore;
    use agent_server::journal::store::{
        AgentTaskStore, CancellationMarkerSink, InMemoryAgentTaskStore,
        InMemoryAtomicSpawnEventCommitter, SubagentInvocationSpawn,
    };
    use agent_server::journal::task::{
        AgentTask, AgentTaskId, LeaseId, SuspensionPayload, WorkerId,
    };
    use agent_server::journal::thread_store::{InMemoryThreadStore, ThreadStore};
    use agent_server::worker::subagent::{
        EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
        InheritedSubagentPolicy, SubagentCapabilityProfile, SubagentSandboxPolicy,
    };
    use agent_server::worker::{
        SubagentResultDeps, SubagentTaskOutcome, execute_subagent_task, resolve_subagent_bootstrap,
    };
    use anyhow::{Context, Result};
    use time::OffsetDateTime;

    use super::t0;

    pub struct Stores {
        pub tasks: InMemoryAgentTaskStore,
        pub threads: InMemoryThreadStore,
        pub messages: InMemoryMessageProjectionStore,
        pub events: InMemoryEventRepository,
    }

    impl Stores {
        pub fn new() -> Self {
            let threads = InMemoryThreadStore::new();
            let events = InMemoryEventRepository::new();
            let outbox = InMemoryOutboxStore::new();
            let messages = InMemoryMessageProjectionStore::new();
            let tasks = InMemoryAgentTaskStore::new()
                .with_cancellation_markers(CancellationMarkerSink {
                    event_repo: Arc::new(events.clone()),
                    outbox_store: Arc::new(outbox.clone()),
                    thread_store: Arc::new(threads.clone()),
                })
                .with_spawn_event_committer(Arc::new(InMemoryAtomicSpawnEventCommitter::new(
                    events.clone(),
                    outbox,
                )))
                .with_message_store(Arc::new(messages.clone()));
            Self {
                tasks,
                threads,
                messages,
                events,
            }
        }

        /// Spawn an invocation under a running parent on `name` and cancel
        /// its child root, so the invocation is runnable.
        pub async fn runnable_invocation(&self, name: &str) -> Result<AgentTask> {
            let parent_thread = ThreadId::from_string(name);
            self.threads.get_or_create(&parent_thread, t0()).await?;
            let parent = AgentTask::new_root_turn(parent_thread.clone(), t0(), 3);
            let parent_id = parent.id.clone();
            self.tasks.submit_root_turn(parent).await?;
            let (worker, lease) = (WorkerId::new(), LeaseId::new());
            self.tasks
                .try_acquire_task(
                    &parent_id,
                    worker.clone(),
                    lease.clone(),
                    t0() + time::Duration::hours(1),
                    t0(),
                )
                .await?
                .context("acquire the parent")?;
            let child_thread = ThreadId::from_string(format!("{name}-child"));
            self.threads.get_or_create(&child_thread, t0()).await?;
            let (_, invocation, child_root) = self
                .tasks
                .spawn_subagent_invocation(
                    &parent_id,
                    &worker,
                    &lease,
                    SubagentInvocationSpawn {
                        child_thread_id: child_thread,
                        spec: spec(),
                        child_root_input: Vec::new(),
                        spawn_index: 0,
                        child_caller_metadata: None,
                        payload: payload(&parent_thread),
                    },
                    Vec::new(),
                    t0(),
                )
                .await?;
            self.tasks.cancel_tree(&child_root.id, t0()).await?;
            Ok(invocation)
        }

        /// The deps that settle an invocation through these stores.
        pub fn deps(&self) -> SubagentResultDeps<'_> {
            SubagentResultDeps {
                task_store: &self.tasks,
                thread_store: &self.threads,
                message_store: &self.messages,
                event_repo: &self.events,
                artifact_store: None,
                completion_hook: None,
            }
        }

        /// Acquire `invocation` at `at` and run it to its settle.
        pub async fn run_invocation(
            &self,
            invocation: &AgentTaskId,
            at: OffsetDateTime,
        ) -> Result<SubagentTaskOutcome> {
            let running = self
                .tasks
                .try_acquire_task(
                    invocation,
                    WorkerId::new(),
                    LeaseId::new(),
                    at + time::Duration::hours(1),
                    at,
                )
                .await?
                .context("acquire the invocation")?;
            let bootstrap = resolve_subagent_bootstrap(running, &self.tasks).await?;
            execute_subagent_task(bootstrap, &self.deps(), at).await
        }
    }

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn spec() -> EffectiveSubagentSpec {
        EffectiveSubagentSpec {
            task: "Change one file".into(),
            prompt: String::new(),
            model: "mock-model".into(),
            max_turns: 4,
            timeout_ms: 30_000,
            depth: 1,
            max_parallel_subagents: 1,
            nickname: None,
            sandbox: SubagentSandboxPolicy::read_only(),
            mcp: EffectiveSubagentMcpPolicy {
                allowed_servers: set(&[]),
            },
            audit_provenance: Some(AuditProvenance::new("mock", "mock-model")),
            inherited_policy: InheritedSubagentPolicy {
                default_model: "mock-model".into(),
                allowed_models: set(&["mock-model"]),
                default_max_turns: 4,
                max_turns: 4,
                default_timeout_ms: 30_000,
                max_timeout_ms: 30_000,
                capability_profiles: BTreeMap::from([(
                    "worker".into(),
                    SubagentCapabilityProfile {
                        capabilities: set(&["read_file"]),
                        sandbox: SubagentSandboxPolicy::read_only(),
                        allowed_mcp_servers: set(&[]),
                    },
                )]),
                allowed_capabilities: set(&["read_file"]),
                max_depth: 3,
                max_parallel_subagents: 1,
                sandbox: SubagentSandboxPolicy::read_only(),
                allowed_mcp_servers: set(&[]),
                audit_provider: "mock".into(),
            },
            capabilities: EffectiveSubagentCapabilities {
                profile: "worker".into(),
                allowed: set(&["read_file"]),
            },
        }
    }

    fn payload(thread: &ThreadId) -> SuspensionPayload {
        let input = serde_json::json!({"task": "Change one file"});
        SuspensionPayload {
            continuation: ContinuationEnvelope::wrap(AgentContinuation {
                thread_id: thread.clone(),
                turn: 1,
                total_usage: TokenUsage::default(),
                turn_usage: TokenUsage::default(),
                pending_tool_calls: vec![PendingToolCallInfo {
                    id: "call_worker".into(),
                    name: "subagent_worker".into(),
                    display_name: "Subagent: Worker".into(),
                    tier: ToolTier::Confirm,
                    concurrency: ToolConcurrency::Exclusive,
                    replay: ToolReplay::Unsafe,
                    input: input.clone(),
                    effective_input: input,
                    listen_context: None,
                }],
                awaiting_index: 0,
                completed_results: Vec::new(),
                state: AgentState::new(thread.clone()),
                response_id: None,
                stop_reason: None,
                response_content: Vec::new(),
            }),
            suspended_messages: Vec::new(),
            child_join_policy: agent_server::ChildJoinPolicy::default(),
        }
    }
}

/// A subagent completion hook's `GitStep` record is counted on
/// `agent_server.effects{kind="git_step"}` (started, then finished), with
/// its run time, and an output the SDK refuses is counted on
/// `agent_server.subagent.completion_hook.refused`.
#[tokio::test]
async fn completion_hook_signals_count_git_steps_and_refusals() -> Result<()> {
    use agent_server::journal::{Settle, Write};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let _ = ServerMetrics::global();

    let stores = hook_fixture::Stores::new();
    let refused_write = Write::SettleTask {
        task: AgentTaskId::new(),
        outcome: Settle::Fail { error: "x".into() },
    };
    for (name, extra_writes) in [
        ("hook-ok", Vec::new()),
        ("hook-refused", vec![refused_write]),
    ] {
        let settled = stores.settle_through_hook(name, extra_writes).await?;
        assert_eq!(
            settled.is_ok(),
            name == "hook-ok",
            "{name}: {:?}",
            settled.err()
        );
    }

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let effects = collect_u64_counter(&snapshots, "agent_server.effects");
    let git_steps = |transition: &str| -> u64 {
        effects
            .iter()
            .filter(|(set, _)| {
                matches_all(
                    set,
                    &[(attrs::KIND, "git_step"), (attrs::TRANSITION, transition)],
                )
            })
            .map(|(_, value)| value)
            .sum()
    };
    assert_eq!(git_steps("started"), 2, "{effects:?}");
    assert_eq!(git_steps("finished"), 1, "{effects:?}");
    let durations = collect_histogram_attrs(&snapshots, "agent_server.effect.duration");
    assert!(
        durations.iter().any(|set| matches_all(
            set,
            &[(attrs::KIND, "git_step"), (attrs::OUTCOME, "completed")]
        )),
        "{durations:?}"
    );
    let refused = collect_u64_counter(&snapshots, "agent_server.subagent.completion_hook.refused");
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert!(matches_all(
        &refused[0].0,
        &[(attrs::REASON, "extra_write")]
    ));
    assert_eq!(refused[0].1, 1);
    Ok(())
}

/// In-memory stores and an invocation fixture for the hook signals test.
mod hook_fixture {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use agent_sdk_foundation::audit::AuditProvenance;
    use agent_sdk_foundation::{
        AgentContinuation, AgentState, ContinuationEnvelope, PendingToolCallInfo, ThreadId,
        TokenUsage, ToolConcurrency, ToolReplay, ToolTier,
    };
    use agent_server::journal::effect::{EffectRecord, EffectState};
    use agent_server::journal::event_repository::InMemoryEventRepository;
    use agent_server::journal::message_store::InMemoryMessageProjectionStore;
    use agent_server::journal::outbox::InMemoryOutboxStore;
    use agent_server::journal::store::SubagentInvocationSpawn;
    use agent_server::journal::store::{
        AgentTaskStore, CancellationMarkerSink, InMemoryAgentTaskStore,
        InMemoryAtomicSpawnEventCommitter,
    };
    use agent_server::journal::task::{AgentTask, LeaseId, SuspensionPayload, WorkerId};
    use agent_server::journal::thread_store::{InMemoryThreadStore, ThreadStore};
    use agent_server::journal::{DurableStore, SpawnChildrenExt, Write};
    use agent_server::worker::subagent::{
        EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
        InheritedSubagentPolicy, SubagentCapabilityProfile, SubagentSandboxPolicy,
    };
    use agent_server::worker::{
        CompletionContext, CompletionOutput, CompletionPlan, InstalledCompletionHook, RunMode,
        SubagentCompletionHook, SubagentResultDeps, execute_subagent_task,
        resolve_subagent_bootstrap,
    };
    use anyhow::{Context, Result};

    use super::t0;

    pub struct Stores {
        pub tasks: InMemoryAgentTaskStore,
        pub threads: InMemoryThreadStore,
        pub messages: InMemoryMessageProjectionStore,
        pub events: InMemoryEventRepository,
    }

    impl Stores {
        pub fn new() -> Self {
            let threads = InMemoryThreadStore::new();
            let events = InMemoryEventRepository::new();
            let outbox = InMemoryOutboxStore::new();
            let messages = InMemoryMessageProjectionStore::new();
            let tasks = InMemoryAgentTaskStore::new()
                .with_cancellation_markers(CancellationMarkerSink {
                    event_repo: Arc::new(events.clone()),
                    outbox_store: Arc::new(outbox.clone()),
                    thread_store: Arc::new(threads.clone()),
                })
                .with_spawn_event_committer(Arc::new(InMemoryAtomicSpawnEventCommitter::new(
                    events.clone(),
                    outbox,
                )))
                .with_message_store(Arc::new(messages.clone()));
            Self {
                tasks,
                threads,
                messages,
                events,
            }
        }

        pub async fn running_parent(
            &self,
            thread: &ThreadId,
            worker: &WorkerId,
            lease: &LeaseId,
        ) -> Result<AgentTask> {
            self.threads.get_or_create(thread, t0()).await?;
            let parent = AgentTask::new_root_turn(thread.clone(), t0(), 3);
            let id = parent.id.clone();
            self.tasks.submit_root_turn(parent).await?;
            self.tasks
                .try_acquire_task(
                    &id,
                    worker.clone(),
                    lease.clone(),
                    t0() + time::Duration::hours(1),
                    t0(),
                )
                .await?
                .context("acquire the parent")
        }

        /// Spawn an invocation of a cancelled child on `name`, and settle it
        /// through a hook whose output carries `extra_writes`.
        pub async fn settle_through_hook(
            &self,
            name: &str,
            extra_writes: Vec<Write>,
        ) -> Result<Result<()>> {
            let parent_thread = ThreadId::from_string(name);
            let (worker, lease) = (WorkerId::new(), LeaseId::new());
            let parent = self.running_parent(&parent_thread, &worker, &lease).await?;
            let child_thread = ThreadId::from_string(format!("{name}-child"));
            agent_server::journal::thread_store::ThreadStore::get_or_create(
                &self.threads,
                &child_thread,
                t0(),
            )
            .await?;
            let (_, invocation, child_root) = self
                .tasks
                .spawn_subagent_invocation(
                    &parent.id,
                    &worker,
                    &lease,
                    SubagentInvocationSpawn {
                        child_thread_id: child_thread,
                        spec: spec(),
                        child_root_input: Vec::new(),
                        spawn_index: 0,
                        child_caller_metadata: None,
                        payload: payload(&parent_thread),
                    },
                    Vec::new(),
                    t0(),
                )
                .await?;
            self.tasks.cancel_tree(&child_root.id, t0()).await?;
            let running = self
                .tasks
                .try_acquire_task(
                    &invocation.id,
                    WorkerId::new(),
                    LeaseId::new(),
                    t0() + time::Duration::hours(1),
                    t0(),
                )
                .await?
                .context("acquire the invocation")?;
            let bootstrap = resolve_subagent_bootstrap(running, &self.tasks).await?;
            let deps = SubagentResultDeps {
                task_store: &self.tasks,
                thread_store: &self.threads,
                message_store: &self.messages,
                event_repo: &self.events,
                artifact_store: None,
                completion_hook: Some(InstalledCompletionHook {
                    hook: Arc::new(Hook { extra_writes }),
                    store: Arc::new(self.tasks.clone()) as Arc<dyn DurableStore>,
                }),
            };
            Ok(execute_subagent_task(bootstrap, &deps, t0())
                .await
                .map(drop))
        }
    }

    struct Hook {
        extra_writes: Vec<Write>,
    }

    #[async_trait::async_trait]
    impl SubagentCompletionHook for Hook {
        fn plan(&self, _ctx: &CompletionContext) -> Option<CompletionPlan> {
            Some(CompletionPlan {
                step: "merge_back".into(),
                repo: "repo_1".into(),
                unit: "unit_1".into(),
            })
        }

        async fn run(
            &self,
            _ctx: &CompletionContext,
            _prior: Option<&EffectRecord>,
            _mode: RunMode,
        ) -> Result<CompletionOutput> {
            Ok(CompletionOutput {
                result: agent_sdk_foundation::ToolResult::success("merged"),
                state: EffectState::Completed,
                extra_writes: self.extra_writes.clone(),
                park: None,
            })
        }
    }

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    pub fn spec() -> EffectiveSubagentSpec {
        EffectiveSubagentSpec {
            task: "Change one file".into(),
            prompt: String::new(),
            model: "mock-model".into(),
            max_turns: 4,
            timeout_ms: 30_000,
            depth: 1,
            max_parallel_subagents: 1,
            nickname: None,
            sandbox: SubagentSandboxPolicy::read_only(),
            mcp: EffectiveSubagentMcpPolicy {
                allowed_servers: set(&[]),
            },
            audit_provenance: Some(AuditProvenance::new("mock", "mock-model")),
            inherited_policy: InheritedSubagentPolicy {
                default_model: "mock-model".into(),
                allowed_models: set(&["mock-model"]),
                default_max_turns: 4,
                max_turns: 4,
                default_timeout_ms: 30_000,
                max_timeout_ms: 30_000,
                capability_profiles: BTreeMap::from([(
                    "worker".into(),
                    SubagentCapabilityProfile {
                        capabilities: set(&["read_file"]),
                        sandbox: SubagentSandboxPolicy::read_only(),
                        allowed_mcp_servers: set(&[]),
                    },
                )]),
                allowed_capabilities: set(&["read_file"]),
                max_depth: 3,
                max_parallel_subagents: 1,
                sandbox: SubagentSandboxPolicy::read_only(),
                allowed_mcp_servers: set(&[]),
                audit_provider: "mock".into(),
            },
            capabilities: EffectiveSubagentCapabilities {
                profile: "worker".into(),
                allowed: set(&["read_file"]),
            },
        }
    }

    pub fn payload(thread: &ThreadId) -> SuspensionPayload {
        let input = serde_json::json!({"task": "Change one file"});
        SuspensionPayload {
            continuation: ContinuationEnvelope::wrap(AgentContinuation {
                thread_id: thread.clone(),
                turn: 1,
                total_usage: TokenUsage::default(),
                turn_usage: TokenUsage::default(),
                pending_tool_calls: vec![PendingToolCallInfo {
                    id: "call_worker".into(),
                    name: "subagent_worker".into(),
                    display_name: "Subagent: Worker".into(),
                    tier: ToolTier::Confirm,
                    concurrency: ToolConcurrency::Exclusive,
                    replay: ToolReplay::Unsafe,
                    input: input.clone(),
                    effective_input: input,
                    listen_context: None,
                }],
                awaiting_index: 0,
                completed_results: Vec::new(),
                state: AgentState::new(thread.clone()),
                response_id: None,
                stop_reason: None,
                response_content: Vec::new(),
            }),
            suspended_messages: Vec::new(),
            child_join_policy: agent_server::ChildJoinPolicy::default(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// tasks.lease_expired carries the reclaim scope
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_lease_reclaim_is_counted_with_its_scope() -> Result<()> {
    use agent_server::journal::{EpochId, LeaseRecoveryScope};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();
    metrics.record_lease_expiry_outcomes(&[RecoveryRecord::requeued(AgentTaskId::from_string(
        "swept",
    ))]);
    metrics.record_lease_reclaim(
        &[
            RecoveryRecord::requeued(AgentTaskId::from_string("reclaimed")),
            RecoveryRecord::failed_closed(
                AgentTaskId::from_string("at-budget"),
                FailureReason::LeaseExpiredBudgetExhausted,
            ),
        ],
        &LeaseRecoveryScope::ForeignEpoch(EpochId::from_string("epoch-b")),
    );

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.tasks.lease_expired");
    let scoped = |scope: &str| -> u64 {
        counts
            .iter()
            .filter(|(s, _)| has_label(s, attrs::SCOPE, scope))
            .map(|(_, v)| v)
            .sum()
    };
    assert_eq!(scoped("expired"), 1, "{counts:?}");
    assert_eq!(scoped("foreign_epoch"), 2, "{counts:?}");

    let reclaims = collect_u64_counter(&snapshots, "agent_server.epoch.reclaims");
    let reclaimed = |scope: &str, action: &str| -> u64 {
        reclaims
            .iter()
            .filter(|(s, _)| {
                has_label(s, attrs::SCOPE, scope) && has_label(s, attrs::ACTION, action)
            })
            .map(|(_, v)| v)
            .sum()
    };
    assert_eq!(reclaimed("expired", "requeued"), 1, "{reclaims:?}");
    assert_eq!(reclaimed("foreign_epoch", "requeued"), 1, "{reclaims:?}");
    assert_eq!(
        reclaimed("foreign_epoch", "failed_interrupted"),
        1,
        "{reclaims:?}"
    );
    Ok(())
}

#[tokio::test]
async fn subagent_stall_signals_count_by_leg_without_the_ids() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let metrics = ServerMetrics::global();

    let stall = |task_id: &'static str, invocation_id: Option<&'static str>| SubagentStallSignal {
        task_id,
        invocation_id,
        traceparent: None,
        budget_ms: 600_000,
    };
    metrics.record_subagent_stall(
        attrs::SUBAGENT_STALL_RUNNING,
        &stall("child-1", Some("invocation-1")),
    );
    metrics.record_subagent_stall(attrs::SUBAGENT_STALL_PARKED, &stall("child-2", None));
    metrics.record_subagent_stall(
        attrs::SUBAGENT_STALL_PARKED,
        &stall("child-3", Some("invocation-3")),
    );

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let stalls = collect_u64_counter(&snapshots, "agent_server.subagent.stalls");
    let by_leg = |leg: &str| -> u64 {
        stalls
            .iter()
            .filter(|(set, _)| matches_all(set, &[(attrs::LEG, leg)]))
            .map(|(_, count)| count)
            .sum()
    };
    assert_eq!(by_leg(attrs::SUBAGENT_STALL_RUNNING), 1);
    assert_eq!(by_leg(attrs::SUBAGENT_STALL_PARKED), 2);
    assert!(
        stalls.iter().all(|(set, _)| set.len() == 1),
        "the counter carries only its bounded leg: {stalls:?}"
    );
    Ok(())
}

struct TextProvider;

#[async_trait::async_trait]
impl agent_sdk_providers::LlmProvider for TextProvider {
    async fn chat(
        &self,
        _request: agent_sdk_foundation::llm::ChatRequest,
    ) -> Result<agent_sdk_foundation::llm::ChatOutcome> {
        use agent_sdk_foundation::llm::{
            ChatOutcome, ChatResponse, ContentBlock, StopReason, Usage,
        };

        Ok(ChatOutcome::Success(ChatResponse {
            id: "metrics-response".into(),
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            model: "metrics-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                served_speed: None,
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                request_input_tokens: None,
            },
        }))
    }

    fn model(&self) -> &'static str {
        "metrics-model"
    }

    fn provider(&self) -> &'static str {
        "metrics"
    }
}

fn text_definition() -> agent_server::worker::definition::AgentDefinition {
    use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};

    AgentDefinition {
        provider: "metrics".into(),
        model: "metrics-model".into(),
        system_prompt: "Answer.".into(),
        max_tokens: 64,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }
}

/// One root turn under its lease, one under a lease it does not hold, on
/// in-memory stores wired as a host wires them.
async fn run_llm_signal_turns() -> Result<()> {
    use agent_server::journal::checkpoint_store::InMemoryCheckpointStore;
    use agent_server::journal::event_notifier::EventNotifier;
    use agent_server::journal::event_repository::InMemoryEventRepository;
    use agent_server::journal::execution_context::build_root_worker_inputs;
    use agent_server::journal::message_store::InMemoryMessageProjectionStore;
    use agent_server::journal::outbox::InMemoryOutboxStore;
    use agent_server::journal::store::{
        AgentTaskStore, CancellationMarkerSink, CompletedTurnSink, InMemoryAgentTaskStore,
    };
    use agent_server::journal::task::{AgentTask, LeaseId, WorkerId};
    use agent_server::journal::thread_store::{InMemoryThreadStore, ThreadStore};
    use agent_server::journal::turn_attempt_store::InMemoryTurnAttemptStore;
    use agent_server::worker::bootstrap::WorkerBootstrapContext;
    use agent_server::worker::root_turn::{RootTurnDeps, execute_root_turn};

    let (threads, events, messages) = (
        InMemoryThreadStore::new(),
        InMemoryEventRepository::new(),
        InMemoryMessageProjectionStore::new(),
    );
    let (attempts, checkpoints) = (
        InMemoryTurnAttemptStore::new(),
        InMemoryCheckpointStore::new(),
    );
    let tasks = InMemoryAgentTaskStore::new()
        .with_cancellation_markers(CancellationMarkerSink {
            event_repo: Arc::new(events.clone()),
            outbox_store: Arc::new(InMemoryOutboxStore::new()),
            thread_store: Arc::new(threads.clone()),
        })
        .with_message_store(Arc::new(messages.clone()))
        .with_completed_turns(CompletedTurnSink {
            attempt_store: Arc::new(attempts.clone()),
            checkpoint_store: Arc::new(checkpoints.clone()),
        });
    let notifier = EventNotifier::new();
    let deps = RootTurnDeps {
        task_store: &tasks,
        thread_store: &threads,
        message_store: &messages,
        attempt_store: &attempts,
        checkpoint_store: &checkpoints,
        event_repo: &events,
        event_notifier: &notifier,
        subagent_spawn_selector: None,
        compaction_config: None,
        compaction_provider: None,
        compaction_artifact_store: None,
        cancel: None,
        wakeup: None,
        activity: None,
        connectivity_waits: None,
    };
    // One root turn under its lease, one under a lease it does not hold.
    for (name, holds_lease) in [("llm-signals-ok", true), ("llm-signals-refused", false)] {
        let thread = ThreadId::from_string(name);
        threads.get_or_create(&thread, t0()).await?;
        let root = AgentTask::new_root_turn(thread.clone(), t0(), 3);
        tasks.submit_root_turn(root.clone()).await?;
        let (worker, lease) = (WorkerId::new(), LeaseId::new());
        let acquired = tasks
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease,
                t0() + time::Duration::hours(1),
                t0(),
            )
            .await?
            .context("acquire the root")?;
        let bootstrap = WorkerBootstrapContext {
            thread_id: thread,
            task_id: root.id.clone(),
            task: acquired.clone(),
            worker_id: worker,
            lease_id: if holds_lease {
                acquired.lease_id.clone().context("lease")?
            } else {
                LeaseId::new()
            },
            definition: text_definition(),
        };
        let inputs =
            build_root_worker_inputs(bootstrap, &threads, &checkpoints, &messages, t0()).await?;
        let outcome = execute_root_turn(
            inputs,
            "hi",
            &TextProvider,
            &deps,
            t0() + time::Duration::seconds(5),
        )
        .await;
        assert_eq!(outcome.is_ok(), holds_lease, "{name}: {:?}", outcome.err());
    }

    Ok(())
}

/// An LLM request's signals: `agent_server.effects{kind="llm_request"}`
/// counts its start, its finish with the completed turn, and a begin the
/// lease guard refused; `agent_server.effect.duration` records the finished
/// request's run time.
#[tokio::test]
async fn llm_request_signals_count_start_finish_and_refusal() -> Result<()> {
    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let _ = ServerMetrics::global();
    run_llm_signal_turns().await?;

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.effects");
    for transition in ["started", "finished", "refused"] {
        let seen: u64 = counts
            .iter()
            .filter(|(set, _)| {
                matches_all(
                    set,
                    &[
                        (attrs::KIND, "llm_request"),
                        (attrs::TRANSITION, transition),
                    ],
                )
            })
            .map(|(_, value)| value)
            .sum();
        assert_eq!(seen, 1, "transition {transition}: {counts:?}");
    }
    let durations = collect_histogram_attrs(&snapshots, "agent_server.effect.duration");
    assert!(
        durations.iter().any(|set| matches_all(
            set,
            &[(attrs::KIND, "llm_request"), (attrs::OUTCOME, "completed")]
        )),
        "{durations:?}"
    );
    Ok(())
}

/// A committed salvage counts `agent_server.effects{kind="llm_request",
/// transition="salvaged"}` once per salvaged request.
#[tokio::test]
async fn a_salvaged_request_is_counted() -> Result<()> {
    use agent_server::journal::effect::EffectKey;
    use agent_server::journal::task::AgentTaskId;
    use agent_server::worker::salvage::{SalvagedRequest, report_salvaged};

    let _guard = acquire_test_lock().await;
    let (provider, exporter) = setup_meter();
    let _ = ServerMetrics::global();
    let owner = AgentTaskId::from_string("task_salvaged");
    report_salvaged(&[SalvagedRequest {
        key: Some(EffectKey::llm(&owner, "attempt_1")),
        owner,
        own: true,
        messages: vec![agent_sdk_foundation::llm::Message::assistant("half")],
    }]);

    flush(&provider)?;
    let snapshots = collected(&exporter)?;
    let counts = collect_u64_counter(&snapshots, "agent_server.effects");
    let salvaged: u64 = counts
        .iter()
        .filter(|(set, _)| {
            matches_all(
                set,
                &[
                    (attrs::KIND, "llm_request"),
                    (attrs::TRANSITION, "salvaged"),
                ],
            )
        })
        .map(|(_, value)| value)
        .sum();
    assert_eq!(salvaged, 1, "{counts:?}");
    Ok(())
}
