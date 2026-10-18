#![cfg(feature = "otel")]

//! `agent_server.root_turn.duration` and `agent_server.root_turn.outcome`:
//! every store records each root turn that reaches a terminal status once,
//! by outcome, on every terminal path; a tool child records nothing.
//!
//! Each path runs on a fresh store of each backend (Postgres when
//! `TEST_DATABASE_URL` is set) and checks the exact points it records, so
//! dropping any store's recording call fails a test on that backend.

use std::sync::Arc;

use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ThreadId, TokenUsage, llm};
use agent_server::journal::SpawnChildrenExt;
use agent_server::journal::checkpoint::CheckpointKind;
use agent_server::journal::commit::{CommitOwnerGuard, CompletedTurnCommit, commit_completed_turn};
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::store::SubagentInvocationSpawn;
use agent_server::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, SuspensionPayload,
    TaskKind, TaskStatus, WorkerId,
};
use agent_server::journal::turn_attempt::{CloseAttemptParams, OpenAttemptParams};
use agent_server::journal::{EpochId, EpochRecord, LeaseRecoveryScope, Settle, Transition, Write};
use agent_server::observability::ServerMetrics;
use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
use agent_server::worker::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
    InheritedSubagentPolicy, SubagentCapabilityProfile, SubagentSandboxPolicy,
};
use agent_service_host::config::{StorageBackend, StorageConfig};
use agent_service_host::stores::StoreRegistry;
use anyhow::{Context, Result, ensure};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use time::{Duration, OffsetDateTime};
use tokio::sync::Mutex;

/// The tests share the global meter provider.
static METER: Mutex<()> = Mutex::const_new(());

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::days(20_000) + Duration::seconds(secs)
}

fn registry_definitions() -> Arc<InMemoryAgentDefinitionRegistry> {
    Arc::new(InMemoryAgentDefinitionRegistry::new(AgentDefinition {
        provider: "anthropic".into(),
        model: "claude-sonnet-4-5-20250929".into(),
        system_prompt: "test".into(),
        max_tokens: 4096,
        tools: Vec::new(),
        thinking: ThinkingPolicy::default(),
        thinking_display: None,
        tools_fn: None,
        tool_input_sanitizer: None,
        request_context_fn: None,
        policy: RuntimePolicy::server_default(),
    }))
}

/// Submit a root created at `at(0)` on `thread` and lease it until
/// `at(lease_until)`.
async fn running_root(
    store: &dyn AgentTaskStore,
    thread: &str,
    max_attempts: u32,
    lease_until: i64,
) -> Result<(AgentTask, WorkerId, LeaseId)> {
    let root = AgentTask::new_root_turn(ThreadId::from_string(thread), at(0), max_attempts);
    let _ = store.submit_root_turn(root.clone()).await?;
    let (worker, lease) = (WorkerId::new(), LeaseId::new());
    let leased = store
        .try_acquire_task(
            &root.id,
            worker.clone(),
            lease.clone(),
            at(lease_until),
            at(0),
        )
        .await?
        .context("the root is acquirable")?;
    Ok((leased, worker, lease))
}

fn suspension(thread: &str) -> SuspensionPayload {
    SuspensionPayload {
        continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
            agent_sdk_foundation::AgentContinuation {
                thread_id: ThreadId::from_string(thread),
                turn: 1,
                total_usage: TokenUsage::default(),
                turn_usage: TokenUsage::default(),
                pending_tool_calls: vec![],
                awaiting_index: 0,
                completed_results: vec![],
                state: agent_sdk_foundation::AgentState::new(ThreadId::from_string(thread)),
                response_id: None,
                stop_reason: None,
                response_content: vec![],
            },
        ),
        suspended_messages: vec![],
        child_join_policy: agent_server::ChildJoinPolicy::default(),
    }
}

/// The worker's normal end of a root turn: a completed-turn commit that
/// settles the root.
async fn complete_through_a_turn_commit(stores: &StoreRegistry, done_at: i64) -> Result<()> {
    let store = stores.task_store.as_ref();
    let (root, worker, lease) = running_root(store, "root-turn-commit", 3, 600).await?;
    let attempt = stores
        .attempt_store
        .open_attempt(OpenAttemptParams {
            task_id: root.id.clone(),
            attempt_number: 1,
            provenance: agent_sdk_foundation::audit::AuditProvenance::new("test", "test-model"),
            request_blob: serde_json::json!({}),
            now: at(1),
            otel_trace_id: None,
            otel_span_id: None,
        })
        .await?;
    let commit = CompletedTurnCommit {
        extra_writes: vec![Write::SettleTask {
            task: root.id.clone(),
            outcome: Settle::CompleteWithoutResult,
        }],
        delivered_injection_ids: Vec::new(),
        checkpoint_kind: CheckpointKind::FullTurn,
        thread_id: root.thread_id.clone(),
        task_id: root.id.clone(),
        expected_turn: 1,
        turn_attempt_id: attempt.id,
        close_attempt_params: CloseAttemptParams {
            response_blob: serde_json::json!({}),
            response_id: None,
            response_model: None,
            stop_reason: Some(llm::StopReason::EndTurn),
            outcome: agent_server::journal::turn_attempt::TurnAttemptOutcome::Success,
            input_tokens: 1,
            output_tokens: 1,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            route_provider: None,
            thinking_mode: None,
            thinking_budget_tokens: None,
            thinking_effort: None,
        },
        messages: vec![llm::Message::assistant("done")],
        turn_usage: TokenUsage::default(),
        agent_state_snapshot: serde_json::json!({}),
        events: vec![AgentEvent::text("m1", "done")],
        outbox_max_attempts: 3,
        owner_guard: Some(CommitOwnerGuard {
            worker_id: worker,
            lease_id: lease,
        }),
        now: at(done_at),
    };
    let outcome = commit_completed_turn(commit, store).await?;
    ensure!(
        outcome
            .settled
            .is_some_and(|(task, _)| task.status == TaskStatus::Completed),
        "the turn commit settles its root"
    );
    Ok(())
}

/// One root turn down each terminal path, plus a tool child that settles:
/// success at 2 s and 3 s, error at 5 s and 11 s, cancelled at 7 s.
async fn end_one_root_turn_down_each_path(stores: &StoreRegistry) -> Result<()> {
    let store = stores.task_store.as_ref();

    complete_through_a_turn_commit(stores, 2).await?;

    let (root, worker, lease) = running_root(store, "root-complete", 3, 600).await?;
    let (done, _) = store
        .complete_task(&root.id, &worker, &lease, at(3))
        .await?;
    ensure!(done.status == TaskStatus::Completed);

    let (root, worker, lease) = running_root(store, "root-fail", 3, 600).await?;
    let (failed, _) = store
        .fail_task(&root.id, &worker, &lease, "boom".into(), at(5))
        .await?;
    ensure!(failed.status == TaskStatus::Failed);

    let (root, _, _) = running_root(store, "root-cancel", 3, 600).await?;
    let cancelled = store.cancel_tree(&root.id, at(7)).await?;
    ensure!(cancelled.transitioned == std::slice::from_ref(&root.id));

    // A parked root's tool child completes; the root resumes, not ends.
    let (root, worker, lease) = running_root(store, "root-tool", 3, 600).await?;
    let (_, children) = store
        .spawn_tool_children(
            &root.id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(3)],
            suspension("root-tool"),
            None,
            Vec::new(),
            at(1),
        )
        .await?;
    let child = children.into_iter().next().context("one tool child")?;
    let (child_worker, child_lease) = (WorkerId::new(), LeaseId::new());
    let _ = store
        .try_acquire_task(
            &child.id,
            child_worker.clone(),
            child_lease.clone(),
            at(600),
            at(2),
        )
        .await?
        .context("the tool child is acquirable")?;
    let (child, parent) = store
        .complete_task(&child.id, &child_worker, &child_lease, at(4))
        .await?;
    ensure!(child.status == TaskStatus::Completed);
    ensure!(parent.is_some_and(|parent| parent.status == TaskStatus::Pending));

    // Its one attempt spent, a root whose lease lapsed fails closed.
    let (root, _, _) = running_root(store, "root-lease", 1, 10).await?;
    let released = store
        .release_expired_leases(at(11), LeaseRecoveryScope::Expired)
        .await?;
    ensure!(
        released.iter().any(|record| record.id == root.id),
        "the sweep took the lapsed root: {released:?}"
    );
    let swept = store.get(&root.id).await?.context("swept root")?;
    ensure!(swept.status == TaskStatus::Failed, "{:?}", swept.status);
    Ok(())
}

/// `(count, sum)` of `agent_server.root_turn.duration` for `outcome`.
fn durations(metrics: &[ResourceMetrics], outcome: &str) -> (u64, f64) {
    let mut found = (0, 0.0);
    for resource in metrics {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != "agent_server.root_turn.duration" {
                    continue;
                }
                let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = metric.data() else {
                    continue;
                };
                for point in histogram.data_points() {
                    if point
                        .attributes()
                        .any(|kv| kv.key.as_str() == "outcome" && kv.value.as_str() == outcome)
                    {
                        found = (found.0 + point.count(), found.1 + point.sum());
                    }
                }
            }
        }
    }
    found
}

/// The `agent_server.root_turn.outcome` count per label.
fn outcomes(metrics: &[ResourceMetrics]) -> Vec<(String, u64)> {
    let mut found = Vec::new();
    for resource in metrics {
        for scope in resource.scope_metrics() {
            for metric in scope.metrics() {
                if metric.name() != "agent_server.root_turn.outcome" {
                    continue;
                }
                let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() else {
                    continue;
                };
                for point in sum.data_points() {
                    let labels: Vec<String> = point
                        .attributes()
                        .map(|kv| format!("{}={}", kv.key.as_str(), kv.value))
                        .collect();
                    found.push((labels.join(","), point.value()));
                }
            }
        }
    }
    found.sort();
    found
}

/// A terminal path one test drives on a fresh store.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// [`end_one_root_turn_down_each_path`].
    Mixed,
    /// `reject_confirmation` ends only a tool child: a root's rejection is
    /// refused, so neither records; the root then fails at 13 s.
    RejectConfirmation,
    /// A root at its budget fails closed at `try_acquire_task` (17 s).
    FailCloseAtTryAcquire,
    /// The same at `acquire_next_runnable` (19 s).
    FailCloseAtAcquireNext,
    /// `cancel_descendants_of_owned` cancels a subagent's child-thread root
    /// (spawned at 2 s, cancelled at 23 s); the running parent records
    /// nothing.
    CancelDescendants,
    /// `cancel_tree` on a parent parked on a subagent cancels it (29 s) and
    /// the child-thread root (27 s).
    ChildRootUnderCancel,
    /// A lease-swept parent at its budget fails closed (31 s) and cancels
    /// the child-thread root (29 s).
    ChildRootUnderFailClose,
    /// A `ForeignEpoch` reclaim fails closed a root at its budget (37 s).
    ForeignEpoch,
    /// A `SupersededEpochs` reclaim does the same (41 s).
    SupersededEpochs,
}

/// `(outcome, count, sum of seconds)`.
type Point = (&'static str, u64, f64);

/// A root on `thread` created at `at(0)`, leased at `at(leased)` until
/// `at(lease_until)`.
async fn lease_root(
    store: &dyn AgentTaskStore,
    thread: &str,
    max_attempts: u32,
    leased: i64,
    lease_until: i64,
) -> Result<(AgentTask, WorkerId, LeaseId)> {
    let root = AgentTask::new_root_turn(ThreadId::from_string(thread), at(0), max_attempts);
    let _ = store.submit_root_turn(root.clone()).await?;
    let (worker, lease) = (WorkerId::new(), LeaseId::new());
    let leased = store
        .try_acquire_task(
            &root.id,
            worker.clone(),
            lease.clone(),
            at(lease_until),
            at(leased),
        )
        .await?
        .context("the root is acquirable")?;
    Ok((leased, worker, lease))
}

/// A pending root whose one attempt is spent: leased once, then released.
async fn exhausted_pending_root(store: &dyn AgentTaskStore, thread: &str) -> Result<AgentTask> {
    let (root, _, _) = lease_root(store, thread, 1, 0, 600).await?;
    store.update(root.clone().release_lease(at(1))?).await?;
    let pending = store.get(&root.id).await?.context("released root")?;
    ensure!(
        pending.status == TaskStatus::Pending && pending.is_budget_exhausted(),
        "{:?} at attempt {}",
        pending.status,
        pending.attempt
    );
    Ok(pending)
}

async fn ensure_status(
    store: &dyn AgentTaskStore,
    id: &AgentTaskId,
    status: TaskStatus,
) -> Result<()> {
    let row = store.get(id).await?.context("row")?;
    ensure!(
        row.status == status,
        "{id}: {:?}, expected {status:?}",
        row.status
    );
    Ok(())
}

fn sample_subagent_spec() -> EffectiveSubagentSpec {
    EffectiveSubagentSpec {
        task: "Inspect durable linkage".into(),
        prompt: "Stay in read-only mode.".into(),
        model: "claude-sonnet-4-5-20250929".into(),
        max_turns: 5,
        timeout_ms: 15_000,
        depth: 1,
        max_parallel_subagents: 1,
        nickname: Some("Scout".into()),
        sandbox: SubagentSandboxPolicy::read_only(),
        mcp: EffectiveSubagentMcpPolicy {
            allowed_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
        },
        audit_provenance: Some(AuditProvenance::new(
            "anthropic",
            "claude-sonnet-4-5-20250929",
        )),
        inherited_policy: InheritedSubagentPolicy {
            default_model: "claude-sonnet-4-5-20250929".into(),
            allowed_models: std::collections::BTreeSet::from([String::from(
                "claude-sonnet-4-5-20250929",
            )]),
            default_max_turns: 5,
            max_turns: 5,
            default_timeout_ms: 15_000,
            max_timeout_ms: 15_000,
            capability_profiles: std::collections::BTreeMap::from([(
                "research".into(),
                SubagentCapabilityProfile {
                    capabilities: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
                    sandbox: SubagentSandboxPolicy::read_only(),
                    allowed_mcp_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
                },
            )]),
            allowed_capabilities: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
            max_depth: 3,
            max_parallel_subagents: 1,
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
            audit_provider: "anthropic".into(),
        },
        capabilities: EffectiveSubagentCapabilities {
            profile: "research".into(),
            allowed: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
        },
    }
}

/// A parent root on `{name}-parent` (created at `at(0)`, leased at
/// `at(1)` until `at(10)`) parked on one subagent spawned at `at(2)`.
/// Returns the parent and the child-thread root.
async fn parent_parked_on_a_subagent(
    stores: &StoreRegistry,
    name: &str,
    max_attempts: u32,
) -> Result<(AgentTask, AgentTask)> {
    let store = stores.task_store.as_ref();
    let parent_thread = format!("{name}-parent");
    let child_thread = ThreadId::from_string(format!("{name}-child"));
    stores
        .thread_store
        .get_or_create(&ThreadId::from_string(&parent_thread), at(0))
        .await?;
    stores
        .thread_store
        .get_or_create(&child_thread, at(0))
        .await?;
    let (parent, worker, lease) = lease_root(store, &parent_thread, max_attempts, 1, 10).await?;
    let (parent, _invocation, child_root) = store
        .spawn_subagent_invocation(
            &parent.id,
            &worker,
            &lease,
            SubagentInvocationSpawn {
                child_thread_id: child_thread,
                spec: sample_subagent_spec(),
                child_root_input: vec![SubmittedInputItem::Text {
                    text: "Inspect durable linkage".into(),
                }],
                spawn_index: 0,
                child_caller_metadata: None,
                payload: suspension(&parent_thread),
            },
            Vec::new(),
            at(2),
        )
        .await?;
    ensure!(
        child_root.kind == TaskKind::RootTurn && child_root.created_at == at(2),
        "{:?} created at {}",
        child_root.kind,
        child_root.created_at
    );
    Ok((parent, child_root))
}

/// Wake `parent` (parked on its children) with a steering note at
/// `at(5)` and lease it at `at(6)` until `at(lease_until)`; the resume
/// spends no attempt.
async fn resume_parent_early(
    store: &dyn AgentTaskStore,
    parent: &AgentTaskId,
    lease_until: i64,
) -> Result<(WorkerId, LeaseId)> {
    let note = vec![llm::ContentBlock::Text {
        text: "status?".into(),
    }];
    store
        .enqueue_steering_resume(parent, note, at(5))
        .await?
        .context("the parked parent wakes for steering")?;
    let (worker, lease) = (WorkerId::new(), LeaseId::new());
    store
        .try_acquire_task(
            parent,
            worker.clone(),
            lease.clone(),
            at(lease_until),
            at(6),
        )
        .await?
        .context("the woken parent is acquirable")?;
    Ok((worker, lease))
}

/// Commit a `BeginEpoch` at `at(secs)`, exclusive when `recovery` is set.
async fn begin_epoch(
    store: &dyn AgentTaskStore,
    secs: i64,
    recovery: Option<&str>,
) -> Result<EpochId> {
    let mut record = EpochRecord::begin(at(secs), Some("root-turn-metrics".to_owned()));
    if let Some(recovery) = recovery {
        record = record.with_recovery(recovery);
    }
    let epoch = record.id.clone();
    store
        .commit(
            Transition {
                guards: Vec::new(),
                writes: vec![Write::BeginEpoch(record)],
            },
            at(secs),
        )
        .await
        .map_err(anyhow::Error::from)?;
    Ok(epoch)
}

/// Fail closed a root at its budget whose lease (to `at(3_600)`) only an
/// epoch reclaim in `scope` takes, at `at(now)`.
async fn reclaim(
    store: &dyn AgentTaskStore,
    root: &AgentTask,
    scope: LeaseRecoveryScope,
    now: i64,
) -> Result<()> {
    let released = store.release_expired_leases(at(now), scope).await?;
    ensure!(
        released.iter().map(|record| &record.id).eq([&root.id]),
        "the reclaim took the root alone: {released:?}"
    );
    ensure_status(store, &root.id, TaskStatus::Failed).await
}

/// [`Path::Mixed`].
async fn mixed(stores: &StoreRegistry) -> Result<Vec<Point>> {
    end_one_root_turn_down_each_path(stores).await?;
    Ok(vec![
        ("cancelled", 1, 7.0),
        ("error", 2, 16.0),
        ("success", 2, 5.0),
    ])
}

/// [`Path::RejectConfirmation`].
async fn reject_confirmation(stores: &StoreRegistry) -> Result<Vec<Point>> {
    let store = stores.task_store.as_ref();
    let (root, worker, lease) = running_root(store, "reject-root", 3, 600).await?;
    store
        .pause_on_confirmation(
            &root.id,
            &worker,
            &lease,
            suspension("reject-root").continuation,
            None,
            at(1),
        )
        .await?;
    let refused = store
        .reject_confirmation(&root.id, "denied".into(), at(2))
        .await;
    ensure!(
        refused.is_err(),
        "a root's rejection is refused: {refused:?}"
    );
    ensure_status(store, &root.id, TaskStatus::AwaitingConfirmation).await?;

    let (root, worker, lease) = running_root(store, "reject-child", 3, 600).await?;
    let (_, children) = store
        .spawn_tool_children(
            &root.id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(3)],
            suspension("reject-child"),
            None,
            Vec::new(),
            at(1),
        )
        .await?;
    let child = children.into_iter().next().context("one tool child")?;
    let (child_worker, child_lease) = (WorkerId::new(), LeaseId::new());
    let _ = store
        .try_acquire_task(
            &child.id,
            child_worker.clone(),
            child_lease.clone(),
            at(600),
            at(2),
        )
        .await?
        .context("the tool child is acquirable")?;
    store
        .pause_on_confirmation(
            &child.id,
            &child_worker,
            &child_lease,
            suspension("reject-child").continuation,
            None,
            at(3),
        )
        .await?;
    let (child, parent) = store
        .reject_confirmation(&child.id, "denied".into(), at(4))
        .await?;
    ensure!(child.status == TaskStatus::Failed);
    ensure!(parent.is_some_and(|parent| parent.status == TaskStatus::Pending));
    let (worker, lease) = (WorkerId::new(), LeaseId::new());
    let _ = store
        .try_acquire_task(&root.id, worker.clone(), lease.clone(), at(600), at(5))
        .await?
        .context("the resumed root is acquirable")?;
    let (failed, _) = store
        .fail_task(&root.id, &worker, &lease, "boom".into(), at(13))
        .await?;
    ensure!(failed.status == TaskStatus::Failed);
    Ok(vec![("error", 1, 13.0)])
}

/// [`Path::FailCloseAtTryAcquire`].
async fn fail_close_at_try_acquire(stores: &StoreRegistry) -> Result<Vec<Point>> {
    let store = stores.task_store.as_ref();
    let root = exhausted_pending_root(store, "acquire-exhausted").await?;
    let acquired = store
        .try_acquire_task(&root.id, WorkerId::new(), LeaseId::new(), at(600), at(17))
        .await?;
    ensure!(acquired.is_none(), "{acquired:?}");
    ensure_status(store, &root.id, TaskStatus::Failed).await?;
    Ok(vec![("error", 1, 17.0)])
}

/// [`Path::FailCloseAtAcquireNext`].
async fn fail_close_at_acquire_next(stores: &StoreRegistry) -> Result<Vec<Point>> {
    let store = stores.task_store.as_ref();
    let root = exhausted_pending_root(store, "scan-exhausted").await?;
    let acquired = store
        .acquire_next_runnable(WorkerId::new(), LeaseId::new(), at(600), at(19))
        .await?;
    ensure!(acquired.is_none(), "{acquired:?}");
    ensure_status(store, &root.id, TaskStatus::Failed).await?;
    Ok(vec![("error", 1, 19.0)])
}

/// [`Path::CancelDescendants`].
async fn cancel_descendants(stores: &StoreRegistry) -> Result<Vec<Point>> {
    let store = stores.task_store.as_ref();
    let (parent, child_root) = parent_parked_on_a_subagent(stores, "descendants", 3).await?;
    let (worker, lease) = resume_parent_early(store, &parent.id, 600).await?;
    let outcome = store
        .cancel_descendants_of_owned(&parent.id, &worker, &lease, at(23))
        .await?;
    ensure!(outcome.transitioned.contains(&child_root.id), "{outcome:?}");
    ensure_status(store, &child_root.id, TaskStatus::Cancelled).await?;
    ensure_status(store, &parent.id, TaskStatus::Running).await?;
    Ok(vec![("cancelled", 1, 21.0)])
}

/// [`Path::ChildRootUnderCancel`].
async fn child_root_under_cancel(stores: &StoreRegistry) -> Result<Vec<Point>> {
    let store = stores.task_store.as_ref();
    let (parent, child_root) = parent_parked_on_a_subagent(stores, "cancel-child", 3).await?;
    let outcome = store.cancel_tree(&parent.id, at(29)).await?;
    ensure!(outcome.transitioned.contains(&parent.id), "{outcome:?}");
    ensure_status(store, &child_root.id, TaskStatus::Cancelled).await?;
    Ok(vec![("cancelled", 2, 56.0)])
}

/// [`Path::ChildRootUnderFailClose`].
async fn child_root_under_fail_close(stores: &StoreRegistry) -> Result<Vec<Point>> {
    let store = stores.task_store.as_ref();
    let (parent, child_root) = parent_parked_on_a_subagent(stores, "fail-close-child", 1).await?;
    let _ = resume_parent_early(store, &parent.id, 10).await?;
    let released = store
        .release_expired_leases(at(31), LeaseRecoveryScope::Expired)
        .await?;
    ensure!(
        released.iter().map(|record| &record.id).eq([&parent.id]),
        "the sweep took the parent alone: {released:?}"
    );
    ensure_status(store, &parent.id, TaskStatus::Failed).await?;
    ensure_status(store, &child_root.id, TaskStatus::Cancelled).await?;
    Ok(vec![("cancelled", 1, 29.0), ("error", 1, 31.0)])
}

/// [`Path::ForeignEpoch`].
async fn foreign_epoch(stores: &StoreRegistry) -> Result<Vec<Point>> {
    let store = stores.task_store.as_ref();
    let (root, _, _) = lease_root(store, "foreign-epoch", 1, 0, 3_600).await?;
    let epoch = begin_epoch(store, 1, None).await?;
    reclaim(store, &root, LeaseRecoveryScope::ForeignEpoch(epoch), 37).await?;
    Ok(vec![("error", 1, 37.0)])
}

/// [`Path::SupersededEpochs`].
async fn superseded_epochs(stores: &StoreRegistry) -> Result<Vec<Point>> {
    let store = stores.task_store.as_ref();
    let _old = begin_epoch(store, 1, Some("postgres_leader")).await?;
    let (root, _, _) = lease_root(store, "superseded-epoch", 1, 2, 3_600).await?;
    let _new = begin_epoch(store, 3, Some("postgres_leader")).await?;
    reclaim(store, &root, LeaseRecoveryScope::SupersededEpochs, 41).await?;
    Ok(vec![("error", 1, 41.0)])
}

/// Drive `path` on `stores` and return the points it must record.
async fn drive(path: Path, stores: &StoreRegistry) -> Result<Vec<Point>> {
    match path {
        Path::Mixed => Box::pin(mixed(stores)).await,
        Path::RejectConfirmation => Box::pin(reject_confirmation(stores)).await,
        Path::FailCloseAtTryAcquire => Box::pin(fail_close_at_try_acquire(stores)).await,
        Path::FailCloseAtAcquireNext => Box::pin(fail_close_at_acquire_next(stores)).await,
        Path::CancelDescendants => Box::pin(cancel_descendants(stores)).await,
        Path::ChildRootUnderCancel => Box::pin(child_root_under_cancel(stores)).await,
        Path::ChildRootUnderFailClose => Box::pin(child_root_under_fail_close(stores)).await,
        Path::ForeignEpoch => Box::pin(foreign_epoch(stores)).await,
        Path::SupersededEpochs => Box::pin(superseded_epochs(stores)).await,
    }
}

/// Drive `path` on `stores` and check that exactly its points were
/// recorded: the count per `outcome`, and each histogram's count and sum.
async fn records_exactly(stores: StoreRegistry, path: Path) -> Result<()> {
    let _meter = METER.lock().await;
    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    ServerMetrics::reset_for_testing();

    stores.initialize().await?;
    let expected = drive(path, &stores).await?;

    provider.force_flush().context("flush metrics")?;
    let metrics = exporter.get_finished_metrics().context("read metrics")?;
    let backend = stores.backend_name();
    let counts: Vec<(String, u64)> = expected
        .iter()
        .map(|(outcome, count, _)| (format!("outcome={outcome}"), *count))
        .collect();
    ensure!(
        outcomes(&metrics) == counts,
        "{backend} {path:?}: {:?}, expected {counts:?}",
        outcomes(&metrics)
    );
    for outcome in ["success", "error", "cancelled"] {
        let (count, sum) = expected
            .iter()
            .find(|(label, _, _)| *label == outcome)
            .map_or((0, 0.0), |(_, count, sum)| (*count, *sum));
        let (recorded, total) = durations(&metrics, outcome);
        ensure!(
            recorded == count && (total - sum).abs() < 1e-6,
            "{backend} {path:?} {outcome}: {recorded} turns, {total} s, expected {count}, {sum} s"
        );
    }
    Ok(())
}

async fn run_in_memory(path: Path) -> Result<()> {
    records_exactly(StoreRegistry::in_memory(registry_definitions()), path).await
}

#[cfg(feature = "sqlite")]
async fn run_sqlite(path: Path) -> Result<()> {
    let file = tempfile::NamedTempFile::new()?;
    let config = StorageConfig {
        backend: StorageBackend::Sqlite {
            path: Some(file.path().display().to_string()),
            max_connections: None,
        },
        ..StorageConfig::default()
    };
    records_exactly(
        StoreRegistry::from_config(&config, registry_definitions())?,
        path,
    )
    .await
}

/// Postgres in a fresh schema when `TEST_DATABASE_URL` is set; otherwise
/// a pass.
#[cfg(feature = "postgres")]
async fn run_postgres(path: Path) -> Result<()> {
    use agent_service_host::config::PostgresStorageConfig;
    use sqlx::Connection;

    let Ok(database_url) = std::env::var("TEST_DATABASE_URL") else {
        return Ok(());
    };
    let schema = format!("root_turn_{}", uuid::Uuid::new_v4().simple());
    let mut admin = sqlx::postgres::PgConnection::connect(&database_url).await?;
    sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
        .execute(&mut admin)
        .await?;
    let config = StorageConfig {
        backend: StorageBackend::Postgres,
        postgres: PostgresStorageConfig {
            database_url: Some(database_url),
            schema: Some(schema.clone()),
            max_connections: 4,
        },
    };
    let result = match StoreRegistry::from_config(&config, registry_definitions()) {
        Ok(stores) => records_exactly(stores, path).await,
        Err(error) => Err(error),
    };
    sqlx::query(sqlx::AssertSqlSafe(format!("DROP SCHEMA {schema} CASCADE")))
        .execute(&mut admin)
        .await?;
    result
}

/// One test per terminal path on the backend `$run` opens.
macro_rules! each_terminal_path {
    ($run:path) => {
        use super::Path;

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn records_each_terminal_root_turn_once() -> anyhow::Result<()> {
            $run(Path::Mixed).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn reject_confirmation_records_no_root_turn() -> anyhow::Result<()> {
            $run(Path::RejectConfirmation).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn fail_close_at_try_acquire_records_an_error() -> anyhow::Result<()> {
            $run(Path::FailCloseAtTryAcquire).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn fail_close_at_acquire_next_runnable_records_an_error() -> anyhow::Result<()> {
            $run(Path::FailCloseAtAcquireNext).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cancel_descendants_of_owned_records_the_child_root() -> anyhow::Result<()> {
            $run(Path::CancelDescendants).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cancel_tree_records_a_child_thread_root() -> anyhow::Result<()> {
            $run(Path::ChildRootUnderCancel).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn fail_close_records_a_child_thread_root() -> anyhow::Result<()> {
            $run(Path::ChildRootUnderFailClose).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn foreign_epoch_reclaim_records_an_error() -> anyhow::Result<()> {
            $run(Path::ForeignEpoch).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn superseded_epoch_reclaim_records_an_error() -> anyhow::Result<()> {
            $run(Path::SupersededEpochs).await
        }
    };
}

mod in_memory {
    each_terminal_path!(super::run_in_memory);
}

#[cfg(feature = "sqlite")]
mod sqlite {
    each_terminal_path!(super::run_sqlite);
}

#[cfg(feature = "postgres")]
mod postgres {
    each_terminal_path!(super::run_postgres);
}
