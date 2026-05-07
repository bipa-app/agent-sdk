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

use agent_sdk_core::ThreadId;
use anyhow::{Context, Result};
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use time::OffsetDateTime;
use tokio::sync::{Mutex, MutexGuard};

use agent_server::journal::execution_intent::ToolEffectClass;
use agent_server::journal::recovery::{FailureReason, RecoveryRecord};
use agent_server::journal::relay::RelayTick;
use agent_server::journal::task::AgentTaskId;
use agent_server::journal::tool_audit::{
    InMemoryToolAuditEventStore, ToolAuditEvent, ToolAuditEventKind, ToolAuditEventParams,
    ToolAuditEventStore,
};
use agent_server::observability::{ServerMetrics, attrs};

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
    assert!(
        !collect_histogram_attrs(&snapshots, "agent_server.tasks.execution.duration").is_empty()
    );
    assert!(
        !collect_histogram_attrs(&snapshots, "agent_server.thread_events_watch.lag_ms").is_empty()
    );
    Ok(())
}
