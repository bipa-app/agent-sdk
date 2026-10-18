#![cfg(all(feature = "sqlite", feature = "otel"))]

//! The signals of a completed turn's extension documents: the extension
//! point's writes and the documents the turn committed.

use std::sync::Arc;

use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::{ThreadId, TokenUsage, llm};
use agent_server::journal::checkpoint::CheckpointKind;
use agent_server::journal::commit::{CompletedTurnCommit, commit_completed_turn};
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, WorkerId};
use agent_server::journal::turn_attempt::{
    CloseAttemptParams, OpenAttemptParams, TurnAttemptId, TurnAttemptOutcome,
};
use agent_server::journal::turn_attempt_store::TurnAttemptStore;
use agent_server::journal::{
    DocExpect, DocKind, DocScope, DocumentKey, DocumentWrite, TransitionExtension, TransitionView,
    Write,
};
use agent_server::observability::{ServerMetrics, attrs};
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, ensure};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use time::OffsetDateTime;

const THREAD: &str = "extension-metrics";

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(secs)
}

/// Books every completed turn into a document on its own thread.
struct BookTurns;

impl TransitionExtension for BookTurns {
    fn extend(&self, view: TransitionView<'_>) -> Vec<Write> {
        let TransitionView::CompletedTurn(commit) = view else {
            return Vec::new();
        };
        let Ok(kind) = DocKind::new("metrics.booked_turn") else {
            return Vec::new();
        };
        vec![Write::Document(DocumentWrite::put(
            DocumentKey::new(DocScope::Thread(commit.thread_id.clone()), kind, ""),
            1,
            DocExpect::Any,
            serde_json::json!({ "turn": commit.expected_turn }),
        ))]
    }
}

/// Adds a write a completed turn may not carry: an event batch, or a
/// document on another thread.
struct Refuse {
    out_of_scope: bool,
}

impl TransitionExtension for Refuse {
    fn extend(&self, view: TransitionView<'_>) -> Vec<Write> {
        let TransitionView::CompletedTurn(commit) = view else {
            return Vec::new();
        };
        if !self.out_of_scope {
            return vec![Write::AppendEvents {
                thread: commit.thread_id.clone(),
                events: Vec::new(),
            }];
        }
        let Ok(kind) = DocKind::new("metrics.booked_turn") else {
            return Vec::new();
        };
        vec![Write::Document(DocumentWrite::put(
            DocumentKey::new(
                DocScope::Thread(ThreadId::from_string("another-thread")),
                kind,
                "",
            ),
            1,
            DocExpect::Any,
            serde_json::json!({}),
        ))]
    }
}

/// Submit and lease a root and open its attempt; return the task and
/// attempt ids its first turn commits with.
async fn start_first_turn(store: &SqliteDurableStore) -> Result<(AgentTaskId, TurnAttemptId)> {
    let thread = ThreadId::from_string(THREAD);
    let root = AgentTask::new_root_turn(thread.clone(), at(1), 3);
    let _ = store.submit_root_turn(root.clone()).await?;
    let _ = store
        .try_acquire_task(&root.id, WorkerId::new(), LeaseId::new(), at(60), at(2))
        .await?;
    let attempt = store
        .open_attempt(OpenAttemptParams {
            task_id: root.id.clone(),
            attempt_number: 1,
            provenance: AuditProvenance::new("crash-test", "crash-test-model"),
            request_blob: serde_json::json!({}),
            now: at(3),
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await?;
    Ok((root.id, attempt.id))
}

fn first_turn(task_id: AgentTaskId, turn_attempt_id: TurnAttemptId) -> CompletedTurnCommit {
    CompletedTurnCommit {
        thread_id: ThreadId::from_string(THREAD),
        task_id,
        expected_turn: 1,
        delivered_injection_ids: Vec::new(),
        turn_attempt_id,
        close_attempt_params: CloseAttemptParams {
            response_blob: serde_json::json!({}),
            response_id: None,
            response_model: None,
            stop_reason: Some(llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 10,
            output_tokens: 5,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            route_provider: None,
            thinking_mode: None,
            thinking_budget_tokens: None,
            thinking_effort: None,
        },
        messages: vec![llm::Message::assistant("done")],
        turn_usage: TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        },
        checkpoint_kind: CheckpointKind::FullTurn,
        agent_state_snapshot: serde_json::json!({}),
        events: Vec::new(),
        outbox_max_attempts: 3,
        owner_guard: None,
        extra_writes: Vec::new(),
        now: at(4),
    }
}

/// The sum of `metric` over the data points whose attributes include every
/// pair in `expected`.
fn counter_total(
    exporter: &InMemoryMetricExporter,
    metric: &str,
    expected: &[(&str, &str)],
) -> Result<u64> {
    let mut total = 0;
    for resource in exporter.get_finished_metrics().context("read metrics")? {
        for scope in resource.scope_metrics() {
            for found in scope.metrics() {
                if found.name() != metric {
                    continue;
                }
                let AggregatedMetrics::U64(MetricData::Sum(sum)) = found.data() else {
                    continue;
                };
                for point in sum.data_points() {
                    let labels: Vec<(String, String)> = point
                        .attributes()
                        .map(|kv| (kv.key.as_str().to_owned(), kv.value.to_string()))
                        .collect();
                    if expected
                        .iter()
                        .all(|(key, value)| labels.iter().any(|(k, v)| k == key && v == value))
                    {
                        total += point.value();
                    }
                }
            }
        }
    }
    Ok(total)
}

#[tokio::test]
async fn a_completed_turn_counts_its_extension_writes_and_documents() -> Result<()> {
    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    ServerMetrics::reset_for_testing();

    let store = SqliteDurableStore::connect("sqlite::memory:")
        .await?
        .with_transition_extension(Arc::new(BookTurns));
    let (task, attempt) = start_first_turn(&store).await?;
    commit_completed_turn(first_turn(task, attempt), &store).await?;

    provider.force_flush().context("flush metrics")?;
    ensure!(
        counter_total(
            &exporter,
            "agent_server.journal.extension.writes",
            &[(
                attrs::EXTENSION_POINT,
                attrs::EXTENSION_POINT_COMPLETED_TURN
            )],
        )? == 1,
        "the extension's completed-turn write is counted"
    );
    ensure!(
        counter_total(
            &exporter,
            "agent_server.journal.documents.written",
            &[
                (attrs::BACKEND, attrs::BACKEND_SQLITE),
                (attrs::SCOPE, "thread"),
                (attrs::OP, attrs::OP_PUT),
            ],
        )? == 1,
        "the completed turn's document is counted"
    );
    Ok(())
}

/// An extension write a completed turn refuses, by kind or by scope, is
/// counted as a refusal and never as a write.
#[tokio::test]
async fn a_refused_completed_turn_extension_write_counts_as_a_refusal() -> Result<()> {
    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    ServerMetrics::reset_for_testing();

    for out_of_scope in [false, true] {
        let store = SqliteDurableStore::connect("sqlite::memory:")
            .await?
            .with_transition_extension(Arc::new(Refuse { out_of_scope }));
        let (task, attempt) = start_first_turn(&store).await?;
        ensure!(
            commit_completed_turn(first_turn(task, attempt), &store)
                .await
                .is_err(),
            "the extension's write refuses the turn (out of scope: {out_of_scope})"
        );
    }

    provider.force_flush().context("flush metrics")?;
    let point = [(
        attrs::EXTENSION_POINT,
        attrs::EXTENSION_POINT_COMPLETED_TURN,
    )];
    let refused = counter_total(&exporter, "agent_server.journal.extension.refused", &point)?;
    let writes = counter_total(&exporter, "agent_server.journal.extension.writes", &point)?;
    ensure!(refused == 2, "both refusals are counted: {refused}");
    ensure!(
        writes == 0,
        "a refused write is not counted as written: {writes}"
    );
    Ok(())
}
