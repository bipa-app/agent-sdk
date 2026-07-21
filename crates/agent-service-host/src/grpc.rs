//! gRPC transport and local-daemon runtime for the durable service host.

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;

use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm::{self, ContentBlock, ContentSource};
use agent_sdk_foundation::{ThreadId, TokenUsage, ToolResult, ToolTier};
use agent_server::journal::checkpoint::NewCheckpointParams;
use agent_server::journal::execution_intent::GuardedExecutionDeps;
use agent_server::journal::fork_transaction::ForkCommitParams;
use agent_server::journal::idempotency::{IdempotencyClaim, IdempotencyKind, IdempotencyRecord};
use agent_server::journal::store::{
    SubmitRootIdempotency, SubmitRootTurnError, SubmitRootTurnParams,
};
use agent_server::journal::task::{
    AgentTask, AgentTaskId, LeaseId, SubmittedInputItem, TaskKind as JournalTaskKind,
    TaskStatus as JournalTaskStatus, WorkerId,
};
use agent_server::journal::task_state::TaskState;
use agent_server::journal::thread_store::{
    ThreadCreation, ThreadCreationOutcome, ThreadIdConflict,
};
use agent_server::worker::{
    ActivityBeacon, AgentDefinitionRegistry, ConfirmationDecision, ConfirmationResumeOutcome,
    apply_confirmation_decision, resolve_tool_bootstrap, resume_confirmed_tool,
};
use anyhow::{Context, Result};
use async_stream::try_stream;
use base64::Engine as _;
use futures::{Stream, StreamExt as _};
use prost::Message as ProstMessage;
use prost_types::{
    Duration as ProtoDuration, ListValue as ProtoListValue, Struct as ProtoStruct,
    Timestamp as ProtoTimestamp, Value as ProtoValue, value::Kind as ProtoValueKind,
};
use time::OffsetDateTime;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

use crate::config::{AdmissionConfig, ServiceConfig};
use crate::health::HealthSurface;
use crate::host::ServiceHost;
use crate::proto::agent::service::v1 as pb;
use crate::proto::agent::service::v1::agent_control_service_server::{
    AgentControlService, AgentControlServiceServer,
};
use crate::proto::agent::service::v1::agent_event_service_server::{
    AgentEventService, AgentEventServiceServer,
};
use crate::runtime::ExecutionRuntime;
use crate::stores::StoreRegistry;

type RpcResult<T> = std::result::Result<T, RpcError>;

#[derive(Debug)]
struct RpcError(Box<Status>);

impl From<Status> for RpcError {
    fn from(status: Status) -> Self {
        Self(Box::new(status))
    }
}

impl From<RpcError> for Status {
    fn from(error: RpcError) -> Self {
        *error.0
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.0.code(), self.0.message())
    }
}

impl std::error::Error for RpcError {}

/// Durable references stored in a `SubmitThreadWork` idempotency
/// record's `result_json` so a retry can reconstruct the response.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SubmitWorkResult {
    thread_id: String,
    task_id: String,
}

/// Durable references for a `CreateThread` idempotency record.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct CreateThreadResult {
    thread_id: String,
}

/// Durable references for a `ForkThread` idempotency record.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ForkThreadResult {
    /// Source thread id the original call forked from.
    source_thread_id: String,
    /// Fork-point turn from the original call.
    fork_after_committed_turns: u32,
    /// The thread id minted by the original call. Returned verbatim on
    /// any retry so clients can treat `ForkThread` as
    /// at-least-once-safe.
    new_thread_id: String,
}

/// Durable references for a `DecideConfirmation` idempotency record.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct DecideConfirmationResult {
    task_id: String,
    parent_task_id: Option<String>,
}

#[derive(Clone)]
struct GrpcShared {
    stores: StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    health: Arc<HealthSurface>,
    shutdown: CancellationToken,
    lease_duration: time::Duration,
    /// The configured worker heartbeat cadence (ADR-0003 I4). Every
    /// `heartbeat_loop` in the process beats at this one interval — including
    /// the detached Confirm-tier drive — so the stall floor
    /// (`2 x heartbeat_interval`) covers every cadence that can persist
    /// activity for a probed subtree.
    heartbeat_interval: std::time::Duration,
    admission: AdmissionConfig,
}

impl GrpcShared {
    const fn new(
        stores: StoreRegistry,
        runtime: Arc<ExecutionRuntime>,
        health: Arc<HealthSurface>,
        shutdown: CancellationToken,
        lease_duration: time::Duration,
        heartbeat_interval: std::time::Duration,
        admission: AdmissionConfig,
    ) -> Self {
        Self {
            stores,
            runtime,
            health,
            shutdown,
            lease_duration,
            heartbeat_interval,
            admission,
        }
    }

    async fn require_thread(&self, thread_id: &ThreadId) -> Result<agent_server::Thread, Status> {
        self.stores
            .thread_store
            .get(thread_id)
            .await
            .map_err(internal_status("loading thread"))?
            .ok_or_else(|| not_found_status("thread", &thread_id.0))
    }

    async fn thread_view(&self, thread_id: &ThreadId) -> Result<pb::ThreadView, Status> {
        let thread = self.require_thread(thread_id).await?;
        let tasks = self
            .stores
            .task_store
            .list_by_thread(thread_id)
            .await
            .map_err(internal_status("listing thread tasks"))?;

        let active_root = tasks
            .iter()
            .filter(|task| task.is_root() && task.status.blocks_root_admission())
            .min_by_key(|task| task.created_at);

        let mut queued_roots: Vec<&AgentTask> = tasks
            .iter()
            .filter(|task| task.is_root() && task.status == JournalTaskStatus::Queued)
            .collect();
        queued_roots.sort_by_key(|task| task.created_at);

        let mut queued_root_snapshots = Vec::with_capacity(queued_roots.len());
        for task in queued_roots {
            queued_root_snapshots.push(self.task_snapshot(task).await?);
        }

        Ok(pb::ThreadView {
            thread: Some(self.thread_snapshot(&thread).await?),
            active_root: match active_root {
                Some(task) => Some(self.task_snapshot(task).await?),
                None => None,
            },
            queued_roots: queued_root_snapshots,
        })
    }

    async fn thread_snapshot(
        &self,
        thread: &agent_server::Thread,
    ) -> Result<pb::ThreadSnapshot, Status> {
        // Read the two boundary sequences with cheap watermark/min
        // queries instead of materialising the entire event journal —
        // this snapshot is embedded in every SubmitThreadWork / GetThread
        // / CreateThread / ForkThread response, so an O(journal-length)
        // read here scales unboundedly with thread age.
        let next_sequence = self
            .stores
            .event_repo
            .next_sequence(&thread.thread_id)
            .await
            .map_err(internal_status("reading thread event watermark"))?;
        let earliest_available_event_sequence = self
            .stores
            .event_repo
            .min_sequence_at_or_after(&thread.thread_id, OffsetDateTime::UNIX_EPOCH)
            .await
            .map_err(internal_status("reading earliest thread event sequence"))?;

        Ok(pb::ThreadSnapshot {
            thread_id: thread.thread_id.0.clone(),
            status: map_thread_status(thread.status),
            committed_turns: thread.committed_turns,
            total_usage: Some(map_token_usage(&thread.total_usage)),
            created_at: Some(map_timestamp(thread.created_at)?),
            updated_at: Some(map_timestamp(thread.updated_at)?),
            latest_event_sequence: next_sequence.checked_sub(1),
            earliest_available_event_sequence,
        })
    }

    async fn message_projection_snapshot(
        &self,
        thread_id: &ThreadId,
    ) -> Result<pb::MessageProjectionSnapshot, Status> {
        self.require_thread(thread_id).await?;
        let now = OffsetDateTime::now_utc();
        let projection = self
            .stores
            .message_store
            .get_or_create(thread_id, now)
            .await
            .map_err(internal_status("loading message projection"))?;

        Ok(pb::MessageProjectionSnapshot {
            thread_id: projection.thread_id.0,
            messages: projection
                .messages
                .iter()
                .map(map_message)
                .collect::<Result<Vec<_>, _>>()?,
            version: projection.version,
            created_at: Some(map_timestamp(projection.created_at)?),
            updated_at: Some(map_timestamp(projection.updated_at)?),
        })
    }

    async fn task_snapshot(&self, task: &AgentTask) -> Result<pb::TaskSnapshot, Status> {
        let state_detail = match &task.state {
            TaskState::None => None,
            TaskState::WaitingOnChildren { child_ids, .. } => Some(
                pb::task_snapshot::StateDetail::WaitingOnChildren(pb::WaitingOnChildrenState {
                    child_task_ids: child_ids.iter().map(ToString::to_string).collect(),
                }),
            ),
            TaskState::SubagentInvocation { invocation } => Some(
                pb::task_snapshot::StateDetail::SubagentInvocation(pb::SubagentInvocationState {
                    child_thread_id: invocation.child_thread_id.to_string(),
                    child_root_task_id: invocation.child_root_task_id.to_string(),
                }),
            ),
            TaskState::ReadyToResume { child_ids, .. } => Some(
                pb::task_snapshot::StateDetail::ReadyToResume(pb::ReadyToResumeState {
                    child_task_ids: child_ids.iter().map(ToString::to_string).collect(),
                }),
            ),
            TaskState::AwaitingConfirmation {
                continuation,
                prepared_operation,
            } => {
                let pending_tool = continuation
                    .payload
                    .pending_tool_calls
                    .get(continuation.payload.awaiting_index)
                    .ok_or_else(|| {
                        Status::internal(format!(
                            "task {} awaiting confirmation without pending tool at index {}",
                            task.id, continuation.payload.awaiting_index
                        ))
                    })?;
                let description = self
                    .resolve_tool_description(task, &pending_tool.name)
                    .await;
                Some(pb::task_snapshot::StateDetail::AwaitingConfirmation(
                    pb::AwaitingConfirmationState {
                        confirmation: Some(pb::PendingConfirmation {
                            task_id: task.id.to_string(),
                            tool_call_id: pending_tool.id.clone(),
                            tool_name: pending_tool.name.clone(),
                            display_name: pending_tool.display_name.clone(),
                            requested_input: Some(map_json_value(&pending_tool.input)?),
                            description,
                            prepared_operation: prepared_operation
                                .as_ref()
                                .map(map_prepared_operation)
                                .transpose()?,
                            paused_at: Some(map_timestamp(task.updated_at)?),
                        }),
                    },
                ))
            }
        };

        Ok(pb::TaskSnapshot {
            task_id: task.id.to_string(),
            thread_id: task.thread_id.0.clone(),
            root_task_id: task.root_id.to_string(),
            parent_task_id: task.parent_id.as_ref().map(ToString::to_string),
            kind: map_task_kind(task.kind),
            status: map_task_status(task.status),
            depth: task.depth,
            attempt: task.attempt,
            max_attempts: task.max_attempts,
            pending_child_count: task.pending_child_count,
            last_error: task.last_error.clone(),
            state_detail,
            created_at: Some(map_timestamp(task.created_at)?),
            updated_at: Some(map_timestamp(task.updated_at)?),
            completed_at: map_optional_timestamp(task.completed_at)?,
        })
    }

    async fn resolve_tool_description(&self, task: &AgentTask, tool_name: &str) -> String {
        let Ok(Some(root)) = self.stores.task_store.get(&task.root_id).await else {
            return tool_name.to_owned();
        };
        let Ok(definition) = self.stores.definition_registry.resolve(&root).await else {
            return tool_name.to_owned();
        };

        definition
            .tools
            .iter()
            .find(|tool| tool.name == tool_name)
            .map_or_else(|| tool_name.to_owned(), |tool| tool.description.clone())
    }

    async fn execute_approved_confirmation(&self, task_id: &AgentTaskId) -> Result<()> {
        let now = OffsetDateTime::now_utc();
        let worker_id = WorkerId::new();
        let lease_id = LeaseId::new();
        let lease_expires_at = now + self.lease_duration;
        let (acquired, _prepared_operation) = self
            .stores
            .task_store
            .approve_confirmation_and_acquire(task_id, worker_id, lease_id, lease_expires_at, now)
            .await
            .context("approving and acquiring confirmation task")?;

        let watermark = self
            .stores
            .event_repo
            .next_sequence(&acquired.thread_id)
            .await
            .context("capturing approved confirmation watermark")?;

        // A Confirm-tier tool runs on its own detached heartbeat (see
        // `drive_approved_confirmation`), so it gets its own beacon. The
        // heartbeat there beats for `bootstrap.task_id` — the same row the
        // collector's emissions describe — which is what makes persisting
        // this beacon on that row correct.
        let bootstrap = resolve_tool_bootstrap(
            acquired,
            self.stores.task_store.as_ref(),
            ActivityBeacon::new(),
        )
        .await
        .context("bootstrapping approved confirmation task")?;

        // Decouple the side-effecting tool from the RPC lifetime. A
        // Confirm-tier tool can run longer than one lease, so driving it
        // inline (a) lost its lease to the sweep mid-flight → re-acquired
        // by a worker → double execution, and (b) was cancelled outright
        // if the gRPC client disconnected (tonic drops the handler
        // future). Run it on a detached task under a heartbeat loop
        // instead; the tool's events land on the live/replay stream and
        // the RPC returns the task in its Running state.
        tokio::spawn(drive_approved_confirmation(DriveApprovedConfirmation {
            stores: self.stores.clone(),
            runtime: Arc::clone(&self.runtime),
            shutdown: self.shutdown.clone(),
            bootstrap,
            watermark,
            lease_duration: self.lease_duration,
            heartbeat_interval: self.heartbeat_interval,
        }));
        Ok(())
    }

    /// Reject an oversized `SubmitThreadWork` input set with a clean
    /// `INVALID_ARGUMENT` (Phase 10 · E). Both an aggregate cap and a
    /// per-item cap are enforced so a single huge inline attachment, or
    /// a flood of medium items, is bounded before anything is stored.
    fn check_submit_input_size(&self, input: &[pb::UserInputItem]) -> Result<(), Status> {
        let mut total: usize = 0;
        for (index, item) in input.iter().enumerate() {
            let item_len = item.encoded_len();
            if let Some(item_cap) = self.admission.max_submit_item_bytes
                && item_len > item_cap
            {
                return Err(Status::invalid_argument(format!(
                    "submit_thread_work input item {index} is {item_len} bytes, \
                     exceeding the per-item limit of {item_cap} bytes"
                )));
            }
            total = total.saturating_add(item_len);
            if let Some(total_cap) = self.admission.max_submit_input_bytes
                && total > total_cap
            {
                return Err(Status::invalid_argument(format!(
                    "submit_thread_work input is at least {total} bytes, \
                     exceeding the aggregate limit of {total_cap} bytes"
                )));
            }
        }
        Ok(())
    }
}

/// Inputs for the detached task that drives an approved confirmation's
/// tool to completion (see [`drive_approved_confirmation`]).
struct DriveApprovedConfirmation {
    stores: StoreRegistry,
    runtime: Arc<ExecutionRuntime>,
    shutdown: CancellationToken,
    bootstrap: agent_server::worker::ToolTaskBootstrap,
    watermark: u64,
    lease_duration: time::Duration,
    heartbeat_interval: std::time::Duration,
}

/// Drive an approved confirmation's resumed tool to completion on a
/// detached task, holding the lease open with a heartbeat loop and
/// publishing the committed tail to live subscribers.
///
/// This is spawned (not awaited) from the `DecideConfirmation` RPC so a
/// long-running Confirm-tier tool neither loses its lease to the sweep
/// nor gets cancelled by a client disconnect.
async fn drive_approved_confirmation(params: DriveApprovedConfirmation) {
    let DriveApprovedConfirmation {
        stores,
        runtime,
        shutdown,
        bootstrap,
        watermark,
        lease_duration,
        heartbeat_interval,
    } = params;

    let task_id = bootstrap.task_id.clone();
    let thread_id = bootstrap.thread_id.clone();
    let worker_id = bootstrap.worker_id.clone();
    let lease_id = bootstrap.lease_id.clone();

    // Per-drive cancellation token (issue #299): a child of the server
    // shutdown token, registered under the tool task id so enforcement
    // paths that cancel this task's row out from under the drive — the
    // subagent deadline sweep, the `CancelTask` RPC — can abort the
    // in-flight tool IMMEDIATELY instead of letting its side effect
    // land after the parent already resumed with a timeout failure.
    // The scope guard deregisters on every exit from this function —
    // including a panic unwinding this detached task — and its nonce
    // check guarantees a wedged predecessor's late drop never strips a
    // successor drive that re-registered under the same task id.
    let drive_cancel = shutdown.child_token();
    let _drive_registration = stores
        .confirm_drive_cancels
        .register(task_id.clone(), drive_cancel.clone());

    // Extend the lease for the lifetime of the tool call. The
    // heartbeat token is rooted at the server shutdown token, so a
    // host shutdown still tears the heartbeat down.
    let heartbeat_cancel = shutdown.child_token();
    let heartbeat_handle = tokio::spawn(crate::host::heartbeat_loop(
        crate::host::HeartbeatLoopParams {
            stores: stores.clone(),
            task_id: task_id.clone(),
            thread_id: thread_id.clone(),
            worker_id: worker_id.clone(),
            lease_id: lease_id.clone(),
            lease_duration,
            // ADR-0003 I4. The CONFIGURED worker cadence, never one derived
            // from the lease. This row's activity is what keeps a PARKED
            // subagent ancestor alive, and that ancestor's stall floor is
            // `2 x heartbeat_interval` — derived from exactly this constant.
            // A drive that beat slower (the old `lease/3`: 20s under a 60s
            // lease, against a floor built from a 5s worker interval) would
            // persist this tool's progress less often than the floor assumes
            // and get an actively-emitting tool reaped. One cadence in the
            // system is what makes the floor correct by construction.
            heartbeat_interval,
            cancel: heartbeat_cancel.clone(),
            // A terminal lease rejection means the row was taken from
            // us (most commonly a `cancel_tree`); trip the drive token
            // so the in-flight tool aborts. This backstops enforcement
            // paths that never consult the drive registry — they still
            // converge within one heartbeat interval.
            task_cancel: drive_cancel.clone(),
            // Not a subagent child-thread root — no stall budget of its
            // own. It still persists activity: this tool's progress is what
            // keeps a PARKED subagent ancestor alive, and that ancestor's
            // stall probe reads this row.
            deadline: crate::host::SubagentDeadlineState::Exempt,
            activity: bootstrap.activity.clone(),
        },
    ));

    // The instant the tool STARTS. Deliberately not called `now`: a
    // Confirm-tier tool is detached precisely because it can run longer than a
    // lease, so by the time it returns this value is stale by the whole
    // execution. It must never be reused as a transition instant (ADR-0003 I3).
    let started_at = OffsetDateTime::now_utc();
    let outcome = resume_confirmed_tool_with_abort_grace(
        &stores,
        &runtime,
        bootstrap,
        &drive_cancel,
        started_at,
    )
    .await;

    heartbeat_cancel.cancel();
    if let Err(join_err) = heartbeat_handle.await {
        warn!(task_id = %task_id, error = %join_err, "approved confirmation heartbeat join failed");
    }

    // Success and policy-denied outcomes need no repair. A
    // force-dropped drive (`None`) is owned by whoever tripped the
    // token (deadline sweep, CancelTask, lease revocation): its row is
    // already terminal, and failing it here would only CAS-reject.
    if let Some(Err(error)) = outcome {
        warn!(?error, task_id = %task_id, "approved confirmation resume failed");
        // ADR-0003 I3: the terminal instant is captured HERE, at the
        // transition — not `started_at`, which predates the whole tool run.
        let failed_at = OffsetDateTime::now_utc();
        if let Err(fail_err) = stores
            .task_store
            .fail_task(
                &task_id,
                &worker_id,
                &lease_id,
                format!("{error:#}"),
                failed_at,
            )
            .await
        {
            warn!(
                task_id = %task_id,
                error = %fail_err,
                "failed to mark approved confirmation task failed after resume error",
            );
        }
    }

    match crate::host::committed_events_from(stores.event_repo.as_ref(), &thread_id, watermark)
        .await
    {
        Ok(events) if !events.is_empty() => stores.event_notifier.notify(&events),
        Ok(_) => {}
        Err(error) => {
            warn!(
                thread_id = %thread_id,
                ?error,
                "loading committed events for approved confirmation publish failed",
            );
        }
    }
}

/// Run the confirmed-tool resume raced against the drive token with
/// the same bounded abort grace the worker pool applies to executions
/// (issue #299): a cooperative tool observes its cancel argument and
/// returns within the grace; a token-blind tool is force-dropped
/// (`None`) — whoever tripped the token already made the task row
/// terminal, so the drop cannot orphan durable state.
async fn resume_confirmed_tool_with_abort_grace(
    stores: &StoreRegistry,
    runtime: &Arc<ExecutionRuntime>,
    bootstrap: agent_server::worker::ToolTaskBootstrap,
    drive_cancel: &CancellationToken,
    now: OffsetDateTime,
) -> Option<Result<ConfirmationResumeOutcome>> {
    let task_id = bootstrap.task_id.clone();
    let thread_id = bootstrap.thread_id.clone();
    let executor_bootstrap = bootstrap.clone();
    let tool_executor = Arc::clone(runtime.tool_executor());
    let exec_cancel = drive_cancel.clone();
    let deps = GuardedExecutionDeps {
        task_store: stores.task_store.as_ref(),
        intent_store: stores.execution_intent_store.as_ref(),
        event_repo: stores.event_repo.as_ref(),
    };

    let mut resume_fut = Box::pin(resume_confirmed_tool(
        bootstrap,
        &deps,
        runtime.confirmation_policy().as_ref(),
        drive_cancel,
        move |_tool_call, collector| {
            let tool_executor = Arc::clone(&tool_executor);
            let executor_bootstrap = executor_bootstrap.clone();
            let cancel = exec_cancel.clone();
            async move {
                tool_executor
                    .execute_tool_call(&executor_bootstrap, collector, cancel)
                    .await
            }
        },
        now,
    ));
    tokio::select! {
        biased;
        outcome = &mut resume_fut => Some(outcome),
        () = drive_cancel.cancelled() => {
            match tokio::time::timeout(crate::host::EXECUTION_ABORT_GRACE, &mut resume_fut).await {
                Ok(outcome) => Some(outcome),
                Err(_elapsed) => {
                    warn!(
                        task_id = %task_id,
                        thread_id = %thread_id,
                        grace_secs = crate::host::EXECUTION_ABORT_GRACE.as_secs(),
                        "approved confirmation drive ignored cancellation past the grace \
                         window; dropping it (its task row is already terminal)",
                    );
                    None
                }
            }
        }
    }
}

#[derive(Clone)]
struct GrpcControlService {
    shared: Arc<GrpcShared>,
}

#[derive(Clone)]
struct GrpcEventService {
    shared: Arc<GrpcShared>,
}

impl GrpcControlService {
    /// Copy the source thread's durable state onto the freshly-minted
    /// destination thread, cut at the chosen turn boundary.
    ///
    /// Returns `(message_count, event_count)` for the response.
    ///
    /// # Pipeline
    ///
    /// 1. Read the source's projection state at the fork boundary
    ///    (the per-turn checkpoint at `turn_number = fork_after`),
    ///    along with every event whose enclosing turn is `<=
    ///    fork_after`. The reads are pure queries — no destination
    ///    state is mutated yet, so a failure here leaves nothing
    ///    behind.
    /// 2. Build a [`ForkCommitParams`] that bundles every write the
    ///    destination needs (thread aggregate, projection,
    ///    checkpoint, events) under one consistent `now` timestamp.
    /// 3. Dispatch to the backend's optional
    ///    [`AtomicForkCommitter`](agent_server::journal::fork_transaction::AtomicForkCommitter)
    ///    if it's exposed on the active [`ThreadStore`]. Durable
    ///    backends (`SQLite`, `Postgres`) wrap the entire write set in
    ///    one transaction so a crash mid-fork can never leave a
    ///    partially-built destination thread visible. Backends that
    ///    don't expose the hook (in-memory) fall back to running the
    ///    same writes sequentially through the per-store trait
    ///    methods — partial state is observable in-process but
    ///    in-memory state evaporates on restart anyway, so the gap
    ///    is closed for production deployments.
    ///
    /// `fork_after == 0` short-circuits steps 1–3 of the read phase
    /// — the destination is created bare (thread aggregate +
    /// empty projection only). Same shape as `CreateThread`.
    async fn copy_thread_state(
        &self,
        source_thread_id: &ThreadId,
        new_thread_id: &ThreadId,
        fork_after: u32,
        creation: Option<ThreadCreation>,
        now: OffsetDateTime,
    ) -> Result<(u64, u64), Status> {
        let params = self
            .build_fork_commit_params(source_thread_id, new_thread_id, fork_after, creation, now)
            .await?;
        let requested_message_count = params.messages.len() as u64;
        let requested_event_count = params.events.len() as u64;

        // Prefer the atomic hook when the backend exposes one. This
        // is the load-bearing case: durable SQL backends commit the
        // entire fork write set under one transaction so
        // mid-operation failures can't leave a half-built
        // destination observable to the next reader.
        let outcome =
            if let Some(committer) = self.shared.stores.thread_store.atomic_fork_committer() {
                committer
                    .commit_fork_atomic(params)
                    .await
                    .map_err(map_thread_creation_error("atomic fork commit"))?
            } else {
                let _guard = match self.shared.stores.thread_store.sequential_fork_lock() {
                    Some(lock) => Some(lock.lock().await),
                    None => None,
                };
                self.commit_fork_sequential(params).await?
            };

        let (message_count, event_count) = match outcome {
            ThreadCreationOutcome::Created => (requested_message_count, requested_event_count),
            ThreadCreationOutcome::Existing => self.fork_destination_counts(new_thread_id).await?,
        };
        Ok((message_count, event_count))
    }

    async fn fork_destination_counts(&self, thread_id: &ThreadId) -> Result<(u64, u64), Status> {
        let message_count = self
            .shared
            .stores
            .message_store
            .get_history(thread_id)
            .await
            .map_err(internal_status("loading existing fork message history"))?
            .len() as u64;
        let event_count = self
            .shared
            .stores
            .event_repo
            .next_sequence(thread_id)
            .await
            .map_err(internal_status("loading existing fork event sequence"))?;
        Ok((message_count, event_count))
    }

    /// Read source state at the fork boundary and assemble the
    /// [`ForkCommitParams`] the atomic committer (or the sequential
    /// fallback) consumes. This phase is read-only on the source —
    /// no destination state is mutated until either commit path
    /// runs.
    async fn build_fork_commit_params(
        &self,
        source_thread_id: &ThreadId,
        new_thread_id: &ThreadId,
        fork_after: u32,
        creation: Option<ThreadCreation>,
        now: OffsetDateTime,
    ) -> Result<ForkCommitParams, Status> {
        if fork_after == 0 {
            // Empty fork — no projection, no checkpoint, no events.
            return Ok(ForkCommitParams {
                new_thread_id: new_thread_id.clone(),
                creation,
                now,
                committed_turns: 0,
                cumulative_total_usage: TokenUsage::default(),
                messages: Vec::new(),
                checkpoint: None,
                events: Vec::new(),
            });
        }

        // Locate the source's checkpoint at the fork boundary.
        let checkpoint = self
            .shared
            .stores
            .checkpoint_store
            .get_by_turn(source_thread_id, fork_after)
            .await
            .map_err(internal_status("loading source checkpoint at fork point"))?
            .ok_or_else(|| {
                Status::failed_precondition(format!(
                    "source thread {} has no checkpoint for committed_turns {} \
                     (the fork point landed on a turn that wasn't durably checkpointed)",
                    source_thread_id.0, fork_after,
                ))
            })?;

        // The source's checkpoint at the fork boundary already
        // carries the cumulative `total_usage` at that turn inside
        // its `agent_state_snapshot`. Pluck it before we rewrite
        // the snapshot so the fork lands at the same total the
        // source reported (instead of starting at zero, which
        // hides the inherited cost).
        let cumulative_total_usage =
            total_usage_from_state_snapshot(&checkpoint.agent_state_snapshot).map_err(
                internal_status("extracting cumulative total_usage from source snapshot"),
            )?;

        // Rewrite the snapshot so the staged state store's
        // bound-thread guard accepts it on the destination.
        let agent_state_snapshot =
            rewrite_state_snapshot_thread_id(&checkpoint.agent_state_snapshot, new_thread_id)
                .map_err(internal_status("rewriting forked agent_state_snapshot"))?;

        // Read source events whose enclosing turn `<= fork_after`.
        let source_events = self
            .shared
            .stores
            .event_repo
            .get_events(source_thread_id)
            .await
            .map_err(internal_status("loading source events for fork"))?;
        let events = events_up_to_fork_boundary(source_events, fork_after);

        Ok(ForkCommitParams {
            new_thread_id: new_thread_id.clone(),
            creation,
            now,
            committed_turns: checkpoint.turn_number,
            cumulative_total_usage,
            messages: checkpoint.messages.clone(),
            checkpoint: Some(NewCheckpointParams {
                // Forking copies the source checkpoint verbatim onto the
                // new thread; its provenance travels with it.
                kind: checkpoint.kind,
                thread_id: new_thread_id.clone(),
                turn_number: checkpoint.turn_number,
                task_id: checkpoint.task_id.clone(),
                messages: checkpoint.messages,
                agent_state_snapshot,
                turn_usage: checkpoint.turn_usage,
                now,
            }),
            events,
        })
    }

    /// Sequential fallback for backends that don't expose
    /// [`AtomicForkCommitter`].
    ///
    /// Mirrors what the atomic path does, but invokes the per-store
    /// trait methods one at a time. Without a transaction, a failure
    /// midway leaves a partially-built thread visible — the in-memory
    /// reference store doesn't care because process restart wipes all
    /// state, and durable backends bypass this fallback by exposing
    /// the atomic hook.
    async fn commit_fork_sequential(
        &self,
        params: ForkCommitParams,
    ) -> Result<ThreadCreationOutcome, Status> {
        // Bootstrap the destination's aggregate + projection rows. The
        // caller-minted path claims its immutable creation parameters under
        // the same thread-store write lock before any projection mutation.
        let outcome = if let Some(creation) = params.creation.as_ref() {
            self.shared
                .stores
                .thread_store
                .get_or_create_for_creation(&params.new_thread_id, creation, params.now)
                .await
                .map_err(map_thread_creation_error("creating forked thread"))?
        } else {
            self.shared
                .stores
                .thread_store
                .get_or_create(&params.new_thread_id, params.now)
                .await
                .map_err(internal_status("creating forked thread"))?;
            ThreadCreationOutcome::Created
        };
        if outcome == ThreadCreationOutcome::Existing {
            return Ok(outcome);
        }
        self.shared
            .stores
            .message_store
            .get_or_create(&params.new_thread_id, params.now)
            .await
            .map_err(internal_status("creating forked message projection"))?;

        if !params.messages.is_empty() {
            self.shared
                .stores
                .message_store
                .replace_history(&params.new_thread_id, params.messages, params.now)
                .await
                .map_err(internal_status("seeding forked message history"))?;
        }

        // Mirror the source's `committed_turns` count + cumulative
        // `total_usage` at the fork boundary. Only the final
        // iteration carries the cumulative usage so the destination
        // ends at exactly that total without distributing arbitrary
        // per-turn values across earlier commits (the per-turn
        // breakdown is preserved on the source's checkpoint chain).
        if params.committed_turns > 0 {
            let zero = TokenUsage::default();
            for turn_index in 0..params.committed_turns {
                let usage_for_this_turn = if turn_index + 1 == params.committed_turns {
                    &params.cumulative_total_usage
                } else {
                    &zero
                };
                self.shared
                    .stores
                    .thread_store
                    .commit_turn(
                        &params.new_thread_id,
                        turn_index + 1,
                        usage_for_this_turn,
                        params.now,
                    )
                    .await
                    .map_err(internal_status(
                        "mirroring source committed_turns onto fork",
                    ))?;
            }
        }

        if let Some(checkpoint) = params.checkpoint {
            self.shared
                .stores
                .checkpoint_store
                .commit_checkpoint(checkpoint)
                .await
                .map_err(internal_status("seeding forked checkpoint"))?;
        }

        if !params.events.is_empty() {
            self.shared
                .stores
                .event_repo
                .commit_event_batch(&params.new_thread_id, params.events, params.now)
                .await
                .map_err(internal_status("re-committing events on forked thread"))?;
        }
        Ok(outcome)
    }

    /// Replay a recorded `ForkThread` outcome under the same
    /// `request_id`. The source's `message_count` / `event_count` are
    /// re-derived from the live stores rather than the original
    /// response — clients that retry minutes after the first call see
    /// a snapshot consistent with the projection right now, not a
    /// stale point-in-time count.
    async fn replay_fork_thread_response(
        &self,
        record: &ForkThreadResult,
    ) -> RpcResult<pb::ForkThreadResponse> {
        let new_thread_id = ThreadId::from_string(&record.new_thread_id);
        let message_count = self
            .shared
            .stores
            .message_store
            .get_history(&new_thread_id)
            .await
            .map_err(internal_status(
                "idempotent fork: load forked message history",
            ))?
            .len() as u64;
        let event_count = self
            .shared
            .stores
            .event_repo
            .next_sequence(&new_thread_id)
            .await
            .map_err(internal_status(
                "idempotent fork: load forked event sequence",
            ))?;
        Ok(pb::ForkThreadResponse {
            thread: Some(self.shared.thread_view(&new_thread_id).await?),
            source_thread_id: record.source_thread_id.clone(),
            message_count,
            event_count,
            fork_after_committed_turns: record.fork_after_committed_turns,
        })
    }

    async fn confirmation_response(
        &self,
        task: &AgentTask,
        parent_task: Option<&AgentTask>,
    ) -> RpcResult<pb::DecideConfirmationResponse> {
        Ok(pb::DecideConfirmationResponse {
            task: Some(self.shared.task_snapshot(task).await?),
            parent_task: match parent_task {
                Some(parent_task) => Some(self.shared.task_snapshot(parent_task).await?),
                None => None,
            },
        })
    }

    async fn replay_decide_confirmation(
        &self,
        record: &DecideConfirmationResult,
    ) -> RpcResult<pb::DecideConfirmationResponse> {
        let task = self
            .shared
            .stores
            .task_store
            .get(&AgentTaskId::from_string(record.task_id.clone()))
            .await
            .map_err(internal_status("loading idempotent confirmation task"))?
            .ok_or_else(|| not_found_status("task", "idempotent confirmation result"))?;
        let parent_task = match &record.parent_task_id {
            Some(parent_task_id) => self
                .shared
                .stores
                .task_store
                .get(&AgentTaskId::from_string(parent_task_id.clone()))
                .await
                .map_err(internal_status("loading idempotent confirmation parent"))?,
            None => None,
        };

        self.confirmation_response(&task, parent_task.as_ref())
            .await
    }

    async fn load_task_and_parent(
        &self,
        task_id: &AgentTaskId,
    ) -> RpcResult<(AgentTask, Option<AgentTask>)> {
        let task = self
            .shared
            .stores
            .task_store
            .get(task_id)
            .await
            .map_err(internal_status("reloading confirmation task"))?
            .ok_or_else(|| not_found_status("task", &task_id.to_string()))?;
        let parent_task = match task.parent_id.as_ref() {
            Some(parent_id) => self
                .shared
                .stores
                .task_store
                .get(parent_id)
                .await
                .map_err(internal_status("reloading confirmation parent"))?,
            None => None,
        };
        Ok((task, parent_task))
    }

    async fn run_confirmation_decision(
        &self,
        request: &pb::DecideConfirmationRequest,
        thread_id: &ThreadId,
        task_id: &AgentTaskId,
    ) -> RpcResult<(AgentTask, Option<AgentTask>)> {
        let current_task = self
            .shared
            .stores
            .task_store
            .get(task_id)
            .await
            .map_err(internal_status("loading confirmation task"))?
            .ok_or_else(|| not_found_status("task", &task_id.to_string()))?;
        if current_task.thread_id != *thread_id {
            return Err(Status::failed_precondition(format!(
                "task {task_id} does not belong to thread {thread_id}",
            ))
            .into());
        }

        let decision = map_confirmation_decision(
            request
                .decision
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("confirmation decision is required"))?,
        )?;
        match decision {
            ConfirmationDecision::Approved => {
                Box::pin(self.shared.execute_approved_confirmation(task_id))
                    .await
                    .map_err(internal_status("executing approved confirmation"))?;
            }
            other => {
                let now = OffsetDateTime::now_utc();
                let _ = apply_confirmation_decision(
                    task_id,
                    other,
                    self.shared.stores.task_store.as_ref(),
                    self.shared.stores.event_repo.as_ref(),
                    now,
                )
                .await
                .map_err(internal_status("applying confirmation decision"))?;
            }
        }

        self.load_task_and_parent(task_id).await
    }
}

#[tonic::async_trait]
impl AgentControlService for GrpcControlService {
    async fn create_thread(
        &self,
        request: Request<pb::CreateThreadRequest>,
    ) -> Result<Response<pb::CreateThreadResponse>, Status> {
        let request = request.into_inner();
        require_request_id(&request.request_id)?;

        if let Some(thread_id) = parse_caller_thread_id(&request.thread_id)? {
            let now = OffsetDateTime::now_utc();
            let _outcome = self
                .shared
                .stores
                .thread_store
                .get_or_create_for_creation(&thread_id, &ThreadCreation::Create, now)
                .await
                .map_err(map_thread_creation_error(
                    "creating caller-addressed thread",
                ))?;
            self.shared
                .stores
                .message_store
                .get_or_create(&thread_id, now)
                .await
                .map_err(internal_status("creating message projection"))?;
            return Ok(Response::new(pb::CreateThreadResponse {
                thread: Some(self.shared.thread_view(&thread_id).await?),
            }));
        }

        // The legacy no-thread-id path remains request-idempotent and keeps
        // the empty fingerprint byte-for-byte. Durable dedup survives restart.
        let fingerprint: &[u8] = &[];
        match self
            .shared
            .stores
            .task_store
            .claim_idempotency(
                &request.request_id,
                IdempotencyKind::CreateThread,
                fingerprint,
            )
            .await
            .map_err(internal_status("create-thread idempotency lookup"))?
        {
            IdempotencyClaim::Conflict => {
                return Err(idempotency_conflict_status(
                    "CreateThread",
                    &request.request_id,
                ));
            }
            IdempotencyClaim::Replay(record) => {
                let result: CreateThreadResult = decode_idempotency_result(&record.result_json)?;
                return Ok(Response::new(pb::CreateThreadResponse {
                    thread: Some(
                        self.shared
                            .thread_view(&ThreadId::from_string(result.thread_id))
                            .await?,
                    ),
                }));
            }
            IdempotencyClaim::Fresh => {}
        }

        let now = OffsetDateTime::now_utc();
        let thread_id = ThreadId::new();
        self.shared
            .stores
            .thread_store
            .get_or_create(&thread_id, now)
            .await
            .map_err(internal_status("creating thread"))?;
        self.shared
            .stores
            .message_store
            .get_or_create(&thread_id, now)
            .await
            .map_err(internal_status("creating message projection"))?;

        let result_json = serde_json::to_value(CreateThreadResult {
            thread_id: thread_id.0.clone(),
        })
        .map_err(|error| {
            Status::internal(format!(
                "encoding create-thread idempotency result: {error}"
            ))
        })?;
        self.shared
            .stores
            .task_store
            .record_idempotency(IdempotencyRecord {
                request_id: request.request_id,
                kind: IdempotencyKind::CreateThread,
                fingerprint: fingerprint.to_vec(),
                result_json,
            })
            .await
            .map_err(internal_status("recording create-thread idempotency"))?;

        Ok(Response::new(pb::CreateThreadResponse {
            thread: Some(self.shared.thread_view(&thread_id).await?),
        }))
    }

    async fn get_thread(
        &self,
        request: Request<pb::GetThreadRequest>,
    ) -> Result<Response<pb::GetThreadResponse>, Status> {
        let request = request.into_inner();
        let thread_id = parse_thread_id(&request.thread_id)?;
        Ok(Response::new(pb::GetThreadResponse {
            thread: Some(self.shared.thread_view(&thread_id).await?),
        }))
    }

    async fn get_thread_messages(
        &self,
        request: Request<pb::GetThreadMessagesRequest>,
    ) -> Result<Response<pb::GetThreadMessagesResponse>, Status> {
        let request = request.into_inner();
        let thread_id = parse_thread_id(&request.thread_id)?;
        Ok(Response::new(pb::GetThreadMessagesResponse {
            projection: Some(self.shared.message_projection_snapshot(&thread_id).await?),
        }))
    }

    async fn list_thread_tasks(
        &self,
        request: Request<pb::ListThreadTasksRequest>,
    ) -> Result<Response<pb::ListThreadTasksResponse>, Status> {
        let request = request.into_inner();
        let thread_id = parse_thread_id(&request.thread_id)?;
        self.shared.require_thread(&thread_id).await?;

        let mut tasks = self
            .shared
            .stores
            .task_store
            .list_by_thread(&thread_id)
            .await
            .map_err(internal_status("listing thread tasks"))?;
        tasks.sort_by_key(|task| (task.created_at, task.depth, task.spawn_index.unwrap_or(0)));

        let mut snapshots = Vec::with_capacity(tasks.len());
        for task in &tasks {
            snapshots.push(self.shared.task_snapshot(task).await?);
        }

        Ok(Response::new(pb::ListThreadTasksResponse {
            tasks: snapshots,
        }))
    }

    async fn get_task(
        &self,
        request: Request<pb::GetTaskRequest>,
    ) -> Result<Response<pb::GetTaskResponse>, Status> {
        let request = request.into_inner();
        let task_id = parse_task_id(&request.task_id)?;
        let task = self
            .shared
            .stores
            .task_store
            .get(&task_id)
            .await
            .map_err(internal_status("loading task"))?
            .ok_or_else(|| not_found_status("task", &task_id.to_string()))?;

        Ok(Response::new(pb::GetTaskResponse {
            task: Some(self.shared.task_snapshot(&task).await?),
        }))
    }

    async fn submit_thread_work(
        &self,
        request: Request<pb::SubmitThreadWorkRequest>,
    ) -> Result<Response<pb::SubmitThreadWorkResponse>, Status> {
        // Capture the inbound W3C `traceparent` (if the gRPC client
        // propagated one) BEFORE `into_inner()` discards the call
        // metadata, and stamp it on the root turn below so the daemon's
        // `invoke_agent` span continues the caller's distributed trace.
        let inbound_traceparent = request
            .metadata()
            .get("traceparent")
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let request = request.into_inner();
        require_request_id(&request.request_id)?;
        let thread_id = parse_thread_id(&request.thread_id)?;

        if request.input.is_empty() {
            return Err(Status::invalid_argument(
                "submit_thread_work requires at least one input item",
            ));
        }

        // Reject oversized input with a clean INVALID_ARGUMENT before
        // doing any durable work, rather than silently truncating or
        // letting a multi-MB attachment land inline in the journal.
        self.shared.check_submit_input_size(&request.input)?;

        let fingerprint = submit_work_fingerprint(&request);

        let thread = self.shared.require_thread(&thread_id).await?;
        if thread.status.is_completed() {
            return Err(Status::failed_precondition(format!(
                "thread {} is completed",
                thread.thread_id
            )));
        }

        let submitted_input = request
            .input
            .iter()
            .map(map_submitted_input_item)
            .collect::<Result<Vec<_>, _>>()?;

        // Opaque per-turn caller metadata (empty string = none). Parsed into a
        // `serde_json::Value` here so a malformed payload is rejected up front
        // with INVALID_ARGUMENT rather than corrupting the durable task record.
        let caller_metadata = if request.caller_metadata.is_empty() {
            None
        } else {
            Some(
                serde_json::from_str::<serde_json::Value>(&request.caller_metadata).map_err(
                    |error| {
                        Status::invalid_argument(format!(
                            "caller_metadata is not valid JSON: {error}"
                        ))
                    },
                )?,
            )
        };

        let now = OffsetDateTime::now_utc();
        // Resolve the agent definition from a throwaway template. The
        // registry resolves on task identity (kind + thread), so the
        // template's retry budget is irrelevant — but we seed it with
        // the reconciled root default so the template can never carry a
        // budget that would fail-closed if it were ever submitted
        // directly (Phase 10 · E).
        let submission_template =
            AgentTask::new_root_turn(thread_id.clone(), now, AgentTask::DEFAULT_ROOT_MAX_ATTEMPTS);
        let definition = self
            .shared
            .stores
            .definition_registry
            .resolve(&submission_template)
            .await
            .map_err(internal_status(
                "resolving root-turn definition for submission",
            ))?;
        let mut task = AgentTask::new_root_turn_with_optional_caller(
            thread_id.clone(),
            submitted_input,
            caller_metadata,
            now,
            definition.policy.max_attempts,
        );
        // The root turn's spans nest under the inbound client span (the
        // worker rebuilds an OTel parent context from this on a fresh
        // turn). `None` when the caller propagated no trace.
        task.otel_traceparent = inbound_traceparent;

        // The idempotency claim + queue-depth cap + admission all run in
        // one store transaction so a retried (restart-surviving) request
        // produces exactly one root turn (no TOCTOU).
        let result_json = serde_json::to_value(SubmitWorkResult {
            thread_id: thread_id.0.clone(),
            task_id: task.id.to_string(),
        })
        .map_err(|error| {
            Status::internal(format!("encoding submit idempotency result: {error}"))
        })?;

        let outcome = self
            .shared
            .stores
            .task_store
            .submit_root_turn_idempotent(SubmitRootTurnParams {
                task,
                idempotency: Some(SubmitRootIdempotency {
                    request_id: request.request_id.clone(),
                    fingerprint,
                    result_json,
                }),
                max_queued_depth: self.shared.admission.max_queued_roots_per_thread,
            })
            .await
            .map_err(|error| {
                map_submit_root_error("SubmitThreadWork", &request.request_id, error)
            })?;

        Ok(Response::new(pb::SubmitThreadWorkResponse {
            thread: Some(self.shared.thread_view(&thread_id).await?),
            task: Some(self.shared.task_snapshot(&outcome.task).await?),
            current_queue_depth: outcome.queued_depth,
        }))
    }

    async fn fork_thread(
        &self,
        request: Request<pb::ForkThreadRequest>,
    ) -> Result<Response<pb::ForkThreadResponse>, Status> {
        let request = request.into_inner();
        require_request_id(&request.request_id)?;
        let source_thread_id = parse_thread_id(&request.source_thread_id)?;
        let caller_thread_id = parse_caller_thread_id(&request.thread_id)?;
        let caller_minted = caller_thread_id.is_some();
        let fork_after = request.fork_after_committed_turns;
        let fingerprint = fork_thread_fingerprint(&request);

        // Request-id replay guards BOTH paths. On the caller-minted path
        // the destination PK claim below stays authoritative for
        // creation, but the request_id remains a true secondary
        // idempotency key: the fingerprint covers the caller's
        // thread_id, so one request_id can never mint two different
        // destinations (it conflicts instead).
        match self
            .shared
            .stores
            .task_store
            .claim_idempotency(
                &request.request_id,
                IdempotencyKind::ForkThread,
                &fingerprint,
            )
            .await
            .map_err(internal_status("fork-thread idempotency lookup"))?
        {
            IdempotencyClaim::Conflict => {
                return Err(idempotency_conflict_status(
                    "ForkThread",
                    &request.request_id,
                ));
            }
            IdempotencyClaim::Replay(record) if caller_minted && record.result_json.is_null() => {
                // A prior caller-minted attempt died between claim and
                // record; fall through — the destination PK claim is
                // idempotent, so re-driving is safe.
            }
            IdempotencyClaim::Replay(record) => {
                let result: ForkThreadResult = decode_idempotency_result(&record.result_json)?;
                let response = self.replay_fork_thread_response(&result).await?;
                return Ok(Response::new(response));
            }
            IdempotencyClaim::Fresh => {}
        }

        // The source must exist before we mint anything. NOT_FOUND
        // here matches `GetThread` / `SubmitThreadWork` semantics for
        // an unknown thread id.
        let source_thread = self.shared.require_thread(&source_thread_id).await?;
        if fork_after > source_thread.committed_turns {
            return Err(Status::invalid_argument(format!(
                "fork_after_committed_turns {} exceeds source's committed_turns {}",
                fork_after, source_thread.committed_turns,
            )));
        }

        let now = OffsetDateTime::now_utc();
        let (new_thread_id, creation) =
            fork_destination(caller_thread_id, &source_thread_id, fork_after);
        let (message_count, event_count) = self
            .copy_thread_state(&source_thread_id, &new_thread_id, fork_after, creation, now)
            .await?;

        let result_json = serde_json::to_value(ForkThreadResult {
            source_thread_id: request.source_thread_id.clone(),
            fork_after_committed_turns: fork_after,
            new_thread_id: new_thread_id.0.clone(),
        })
        .map_err(|error| {
            Status::internal(format!("encoding fork-thread idempotency result: {error}"))
        })?;
        self.shared
            .stores
            .task_store
            .record_idempotency(IdempotencyRecord {
                request_id: request.request_id.clone(),
                kind: IdempotencyKind::ForkThread,
                fingerprint,
                result_json,
            })
            .await
            .map_err(internal_status("recording fork-thread idempotency"))?;

        Ok(Response::new(pb::ForkThreadResponse {
            thread: Some(self.shared.thread_view(&new_thread_id).await?),
            source_thread_id: request.source_thread_id,
            message_count,
            event_count,
            fork_after_committed_turns: fork_after,
        }))
    }

    async fn decide_confirmation(
        &self,
        request: Request<pb::DecideConfirmationRequest>,
    ) -> Result<Response<pb::DecideConfirmationResponse>, Status> {
        let request = request.into_inner();
        require_request_id(&request.request_id)?;
        let thread_id = parse_thread_id(&request.thread_id)?;
        let task_id = parse_task_id(&request.task_id)?;
        let fingerprint = decide_confirmation_fingerprint(&request);

        // Durable idempotency replay (Phase 10 · E): a retried decision
        // — even across a restart — applies the decision exactly once.
        match self
            .shared
            .stores
            .task_store
            .claim_idempotency(
                &request.request_id,
                IdempotencyKind::DecideConfirmation,
                &fingerprint,
            )
            .await
            .map_err(internal_status("decide-confirmation idempotency lookup"))?
        {
            IdempotencyClaim::Conflict => {
                return Err(idempotency_conflict_status(
                    "DecideConfirmation",
                    &request.request_id,
                ));
            }
            IdempotencyClaim::Replay(record) => {
                let result: DecideConfirmationResult =
                    decode_idempotency_result(&record.result_json)?;
                return Ok(Response::new(
                    self.replay_decide_confirmation(&result).await?,
                ));
            }
            IdempotencyClaim::Fresh => {}
        }

        let (task, parent_task) = self
            .run_confirmation_decision(&request, &thread_id, &task_id)
            .await?;

        let result_json = serde_json::to_value(DecideConfirmationResult {
            task_id: task.id.to_string(),
            parent_task_id: parent_task.as_ref().map(|parent| parent.id.to_string()),
        })
        .map_err(|error| {
            Status::internal(format!(
                "encoding decide-confirmation idempotency result: {error}"
            ))
        })?;
        self.shared
            .stores
            .task_store
            .record_idempotency(IdempotencyRecord {
                request_id: request.request_id,
                kind: IdempotencyKind::DecideConfirmation,
                fingerprint,
                result_json,
            })
            .await
            .map_err(internal_status("recording decide-confirmation idempotency"))?;

        Ok(Response::new(
            self.confirmation_response(&task, parent_task.as_ref())
                .await?,
        ))
    }

    /// Cancel a root task and its entire descendant subtree.
    ///
    /// Interior (non-root) task ids are rejected with
    /// `FAILED_PRECONDITION`: on the durable SQL backends `cancel_tree`
    /// does not rebalance an out-of-subtree parent left in
    /// `WaitingOnChildren`, so cancelling an interior task would strand
    /// that parent forever. Restricting the RPC to roots avoids that
    /// liveness deadlock until the rebalance lands.
    ///
    /// The `reason` is recorded in the server logs only; it is not
    /// persisted on the task row.
    ///
    /// The terminal `cancelled` marker (one per affected thread,
    /// child threads included) is committed inside the cancellation
    /// transaction by the store, with an outbox advisory for
    /// cross-host followers. Cancelling a `Running` root is still
    /// asynchronous for its worker (post-marker salvage, slot-shifted
    /// successor commits) — see the `CancelTask` proto docs and
    /// [`agent_server::worker::cancel_root_turn`].
    async fn cancel_task(
        &self,
        request: Request<pb::CancelTaskRequest>,
    ) -> Result<Response<pb::CancelTaskResponse>, Status> {
        let request = request.into_inner();
        let task_id = parse_task_id(&request.task_id)?;

        // Resolve the task first so an unknown id surfaces a clean
        // NOT_FOUND (matching `GetTask`) instead of the INTERNAL that
        // `cancel_tree`'s own existence-check error would map to. The
        // loaded row also carries the thread id for the cancellation log.
        let task = self
            .shared
            .stores
            .task_store
            .get(&task_id)
            .await
            .map_err(internal_status("loading task for cancellation"))?
            .ok_or_else(|| not_found_status("task", &task_id.to_string()))?;

        // Thread guard: reject a `(thread_id, task_id)` pair whose task
        // belongs to a different thread rather than cancelling some other
        // thread's work. An empty `thread_id` skips the guard.
        if !request.thread_id.trim().is_empty() {
            let thread_id = parse_thread_id(&request.thread_id)?;
            if task.thread_id != thread_id {
                return Err(Status::failed_precondition(format!(
                    "task {task_id} does not belong to thread {thread_id}",
                )));
            }
        }

        // Restrict cancellation to root turns. `cancel_tree` on an
        // interior task cancels that subtree but leaves an out-of-subtree
        // parent stuck in `WaitingOnChildren` on the durable SQL backends
        // — a liveness deadlock this RPC would make client-triggerable.
        if !task.kind.is_root() {
            return Err(Status::failed_precondition(
                "CancelTask only supports root tasks; cancelling an interior task is not yet supported",
            ));
        }

        info!(
            task_id = %task_id,
            thread_id = %task.thread_id,
            reason = %bounded_log_str(&request.reason, MAX_CANCEL_REASON_LOG_CHARS),
            "cancelling task tree",
        );

        // Route through the worker's cancel lifecycle, not bare
        // `cancel_tree`: for a root parked in `WaitingOnChildren` (or
        // `Pending` + `ReadyToResume`) there is no live worker to salvage
        // the turn, so `cancel_root_turn` first commits the completed
        // prefix of the parked turn and closes open attempts — otherwise
        // `cancel_tree` clears the durable continuation and the next turn
        // resumes from stale conversation history. A `Running` root is
        // still handled by its worker (lease loss trips the cancel token
        // and seam B commits the prefix).
        let now = OffsetDateTime::now_utc();
        let cancelled = agent_server::worker::cancel_root_turn(
            &task_id,
            &self.shared.stores.root_turn_deps(),
            now,
        )
        .await
        .map_err(internal_status("cancelling task tree"))?;

        // Trip any live approved-confirmation drive whose tool task was
        // just cancelled (issue #299): without this, the detached drive
        // only notices at its next heartbeat rejection and its side
        // effect could still land post-cancel. Ids without a live
        // drive are no-ops.
        for cancelled_id in &cancelled {
            self.shared
                .stores
                .confirm_drive_cancels
                .cancel(cancelled_id);
        }

        let cancelled_task_ids: Vec<String> = cancelled.iter().map(ToString::to_string).collect();
        let cancelled_count = u32::try_from(cancelled_task_ids.len())
            .map_err(|_| Status::internal("cancelled task count out of range"))?;

        Ok(Response::new(pb::CancelTaskResponse {
            cancelled_task_ids,
            cancelled_count,
        }))
    }
}

type EventStream =
    Pin<Box<dyn Stream<Item = Result<pb::StreamThreadEventsResponse, Status>> + Send>>;

struct ReplayState {
    earliest_available: Option<u64>,
    latest_available: Option<u64>,
    next_sequence: u64,
}

impl GrpcEventService {
    fn build_event_stream(
        &self,
        thread_id: ThreadId,
        after_sequence: Option<u64>,
        follow_mode: pb::FollowMode,
    ) -> EventStream {
        let shared = Arc::clone(&self.shared);
        let stream = try_stream! {
            // Subscribe before reading the journal watermark so no event
            // committed during replay setup slips through the gap between
            // the read and the live subscription.
            let live_rx = shared.stores.event_notifier.subscribe(&thread_id);
            let replay = load_replay_state(shared.as_ref(), &thread_id).await?;
            if has_retention_gap(after_sequence, replay.earliest_available) {
                yield retention_gap_response(
                    &thread_id,
                    after_sequence,
                    replay.earliest_available,
                    replay.latest_available,
                );
                return;
            }

            yield replay_opened_response(
                &thread_id,
                after_sequence,
                replay.earliest_available,
                replay.next_sequence.checked_sub(1),
                follow_mode,
            );

            // Fetch only the suffix the client needs: a bounded range
            // query for a reconnect near the head, the full history only
            // for replay-from-start. The retention-gap check above ran on
            // cheap watermark/min metadata, so a below-floor reconnect
            // never pays for a journal read it would discard.
            let replay_events = fetch_replay_events(
                shared.as_ref(),
                &thread_id,
                after_sequence,
                replay.latest_available,
            )
            .await?;
            let mut last_delivered_sequence = after_sequence;
            for event in &replay_events {
                last_delivered_sequence = Some(event.sequence);
                yield event_stream_response(event)?;
            }

            yield replay_catchup_complete_response(
                &thread_id,
                replay_events.last().map(|event| event.sequence).or(after_sequence),
                follow_mode,
            );

            let last_replayed_terminal = match replay_events.last() {
                Some(event) => confirmed_close_reason(&shared, &thread_id, event).await,
                None => None,
            };
            if let Some(reason) = post_replay_close_reason(follow_mode, last_replayed_terminal) {
                yield closed_stream_response(&thread_id, reason, last_delivered_sequence);
                return;
            }

            // Re-yield the live follow phase. Splitting it into its own
            // stream keeps the yields (and their exact ordering) identical
            // while moving the loop body out of this function.
            let live = follow_live_events(
                Arc::clone(&shared),
                thread_id.clone(),
                live_rx,
                last_delivered_sequence,
            );
            let mut live = std::pin::pin!(live);
            while let Some(item) = live.next().await {
                yield item?;
            }
        };

        Box::pin(stream)
    }
}

/// Close reason for a stream once replay catch-up finishes, or `None`
/// when the stream should transition to the live follow phase.
///
/// `last_replayed_terminal` is the close reason derived from the last
/// replayed event ([`terminal_close_reason`]): a replay that ends on a
/// terminal lifecycle event (`Done` / `Cancelled`) closes instead of
/// following a thread whose work already finished.
const fn post_replay_close_reason(
    follow_mode: pb::FollowMode,
    last_replayed_terminal: Option<pb::StreamCloseReason>,
) -> Option<pb::StreamCloseReason> {
    if matches!(follow_mode, pb::FollowMode::ReplayOnly) {
        Some(pb::StreamCloseReason::ReplayExhausted)
    } else {
        last_replayed_terminal
    }
}

/// Live follow phase: deliver newly-committed events in order, backfilling
/// any gap the unordered broadcast skipped, until the thread completes,
/// the subscriber lags, or the server shuts down.
fn follow_live_events(
    shared: Arc<GrpcShared>,
    thread_id: ThreadId,
    mut live_rx: agent_server::EventReceiver,
    mut last_delivered_sequence: Option<u64>,
) -> impl Stream<Item = Result<pb::StreamThreadEventsResponse, Status>> + Send {
    try_stream! {
        loop {
            tokio::select! {
                () = shared.shutdown.cancelled() => {
                    yield closed_stream_response(
                        &thread_id,
                        pb::StreamCloseReason::ServerShutdown,
                        last_delivered_sequence,
                    );
                    return;
                }
                received = live_rx.recv() => match received {
                    Ok(event) => {
                        // Already delivered (duplicate or stale broadcast).
                        if last_delivered_sequence.is_some_and(|sequence| event.sequence <= sequence) {
                            continue;
                        }
                        // Gap detection: publishers notify the broadcast
                        // channel without a shared ordering lock, so a
                        // later range can arrive before an earlier one on
                        // a thread with parallel writers. If this event
                        // skips ahead of the contiguous frontier, re-read
                        // the missed range from the durable journal and
                        // deliver it in order before the out-of-order
                        // event — otherwise those sequences are dropped
                        // permanently until the client reconnects.
                        let expected_next =
                            last_delivered_sequence.map_or(0, |sequence| sequence + 1);
                        if event.sequence > expected_next {
                            let Ok(missed) = fetch_missed_range(
                                shared.as_ref(),
                                &thread_id,
                                last_delivered_sequence,
                                event.sequence,
                            )
                            .await
                            else {
                                // Can't backfill the gap from the
                                // journal; force the client to
                                // reconnect and replay so no sequence
                                // is silently lost.
                                yield replay_required_response(
                                    &thread_id,
                                    last_delivered_sequence,
                                );
                                return;
                            };
                            for missed_event in &missed {
                                if last_delivered_sequence
                                    .is_some_and(|sequence| missed_event.sequence <= sequence)
                                {
                                    continue;
                                }
                                last_delivered_sequence = Some(missed_event.sequence);
                                let close_reason =
                                    confirmed_close_reason(&shared, &thread_id, missed_event)
                                        .await;
                                yield event_stream_response(missed_event)?;
                                if let Some(reason) = close_reason {
                                    yield closed_stream_response(
                                        &thread_id,
                                        reason,
                                        last_delivered_sequence,
                                    );
                                    return;
                                }
                            }
                        }
                        last_delivered_sequence = Some(event.sequence);
                        let close_reason =
                            confirmed_close_reason(&shared, &thread_id, &event).await;
                        yield event_stream_response(&event)?;
                        if let Some(reason) = close_reason {
                            yield closed_stream_response(
                                &thread_id,
                                reason,
                                last_delivered_sequence,
                            );
                            return;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        yield replay_required_response(&thread_id, last_delivered_sequence);
                        return;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        yield closed_stream_response(
                            &thread_id,
                            pb::StreamCloseReason::ServerShutdown,
                            last_delivered_sequence,
                        );
                        return;
                    }
                }
            }
        }
    }
}

#[tonic::async_trait]
impl AgentEventService for GrpcEventService {
    type StreamThreadEventsStream = EventStream;

    async fn stream_thread_events(
        &self,
        request: Request<pb::StreamThreadEventsRequest>,
    ) -> Result<Response<Self::StreamThreadEventsStream>, Status> {
        let request = request.into_inner();
        let thread_id = parse_thread_id(&request.thread_id)?;
        self.shared.require_thread(&thread_id).await?;
        let follow_mode = map_follow_mode(request.follow_mode)?;
        Ok(Response::new(self.build_event_stream(
            thread_id,
            request.after_sequence,
            follow_mode,
        )))
    }
}

async fn load_replay_state(shared: &GrpcShared, thread_id: &ThreadId) -> RpcResult<ReplayState> {
    // Cheap boundary metadata only — the actual replay events are
    // fetched (bounded) after the retention-gap check passes so a
    // below-floor reconnect never materialises the journal.
    let next_sequence = shared
        .stores
        .event_repo
        .next_sequence(thread_id)
        .await
        .map_err(as_stream_status("capturing event watermark"))?;
    let earliest_available = shared
        .stores
        .event_repo
        .min_sequence_at_or_after(thread_id, OffsetDateTime::UNIX_EPOCH)
        .await
        .map_err(as_stream_status("reading earliest available event"))?;

    Ok(ReplayState {
        earliest_available,
        latest_available: next_sequence.checked_sub(1),
        next_sequence,
    })
}

/// Fetch the events a freshly-opened stream must replay.
///
/// `after_sequence = Some(n)` is the reconnect-near-the-head case: only
/// the suffix `(n, watermark]` is read via `get_events_in_range`, so a
/// client resuming at sequence 10,000 does not reload and discard 10,000
/// rows. `None` (replay-from-start) is the only path that reads the full
/// history.
async fn fetch_replay_events(
    shared: &GrpcShared,
    thread_id: &ThreadId,
    after_sequence: Option<u64>,
    latest_available: Option<u64>,
) -> Result<Vec<agent_server::CommittedEvent>, Status> {
    match after_sequence {
        Some(after) => {
            let up_to = latest_available.unwrap_or(0);
            shared
                .stores
                .event_repo
                .get_events_in_range(thread_id, after, up_to)
                .await
                .map_err(as_stream_status("loading replay events in range"))
        }
        None => shared
            .stores
            .event_repo
            .get_events(thread_id)
            .await
            .map_err(as_stream_status("loading replay events")),
    }
}

/// Re-read the contiguous range of events the live broadcast skipped,
/// `(last_delivered, incoming_sequence)` exclusive on both ends, so an
/// out-of-order delivery never silently drops the events between the
/// frontier and the event that overtook them.
async fn fetch_missed_range(
    shared: &GrpcShared,
    thread_id: &ThreadId,
    last_delivered: Option<u64>,
    incoming_sequence: u64,
) -> Result<Vec<agent_server::CommittedEvent>, Status> {
    let up_to = incoming_sequence.saturating_sub(1);
    match last_delivered {
        // Bounded range read of the gap.
        Some(after) => shared
            .stores
            .event_repo
            .get_events_in_range(thread_id, after, up_to)
            .await
            .map_err(as_stream_status("re-reading missed live range")),
        // Nothing delivered yet: the gap starts at sequence 0, which the
        // exclusive-lower range query cannot express. The journal is at
        // most `incoming_sequence` events long here, so a bounded full
        // read filtered to the prefix is cheap.
        None => Ok(shared
            .stores
            .event_repo
            .get_events(thread_id)
            .await
            .map_err(as_stream_status("re-reading missed live prefix"))?
            .into_iter()
            .filter(|event| event.sequence < incoming_sequence)
            .collect()),
    }
}

const fn has_retention_gap(after_sequence: Option<u64>, earliest_available: Option<u64>) -> bool {
    matches!(
        (after_sequence, earliest_available),
        (Some(requested_after), Some(earliest)) if requested_after < earliest.saturating_sub(1)
    )
}

fn retention_gap_response(
    thread_id: &ThreadId,
    requested_after_sequence: Option<u64>,
    earliest_available_sequence: Option<u64>,
    latest_available_sequence: Option<u64>,
) -> pb::StreamThreadEventsResponse {
    pb::StreamThreadEventsResponse {
        item: Some(pb::stream_thread_events_response::Item::RetentionGap(
            pb::RetentionGap {
                thread_id: thread_id.0.clone(),
                requested_after_sequence,
                earliest_available_sequence,
                latest_available_sequence,
            },
        )),
    }
}

fn replay_opened_response(
    thread_id: &ThreadId,
    requested_after_sequence: Option<u64>,
    earliest_available_sequence: Option<u64>,
    replay_through_sequence: Option<u64>,
    follow_mode: pb::FollowMode,
) -> pb::StreamThreadEventsResponse {
    pb::StreamThreadEventsResponse {
        item: Some(pb::stream_thread_events_response::Item::ReplayOpened(
            pb::ReplayOpened {
                thread_id: thread_id.0.clone(),
                requested_after_sequence,
                earliest_available_sequence,
                replay_through_sequence,
                follow_mode: follow_mode as i32,
            },
        )),
    }
}

fn replay_catchup_complete_response(
    thread_id: &ThreadId,
    replayed_through_sequence: Option<u64>,
    follow_mode: pb::FollowMode,
) -> pb::StreamThreadEventsResponse {
    pb::StreamThreadEventsResponse {
        item: Some(
            pb::stream_thread_events_response::Item::ReplayCatchupComplete(
                pb::ReplayCatchupComplete {
                    thread_id: thread_id.0.clone(),
                    replayed_through_sequence,
                    following_live: follow_mode == pb::FollowMode::ReplayAndFollow,
                },
            ),
        ),
    }
}

fn closed_stream_response(
    thread_id: &ThreadId,
    reason: pb::StreamCloseReason,
    last_sequence: Option<u64>,
) -> pb::StreamThreadEventsResponse {
    pb::StreamThreadEventsResponse {
        item: Some(pb::stream_thread_events_response::Item::Closed(
            pb::EventStreamClosed {
                thread_id: thread_id.0.clone(),
                reason: reason as i32,
                last_sequence,
            },
        )),
    }
}

fn replay_required_response(
    thread_id: &ThreadId,
    last_delivered_sequence: Option<u64>,
) -> pb::StreamThreadEventsResponse {
    pb::StreamThreadEventsResponse {
        item: Some(pb::stream_thread_events_response::Item::ReplayRequired(
            pb::ReplayRequired {
                thread_id: thread_id.0.clone(),
                last_delivered_sequence,
                reason: pb::ReplayRequiredReason::SubscriberLagged as i32,
            },
        )),
    }
}

fn event_stream_response(
    event: &agent_server::CommittedEvent,
) -> RpcResult<pb::StreamThreadEventsResponse> {
    Ok(pb::StreamThreadEventsResponse {
        item: Some(pb::stream_thread_events_response::Item::Event(
            map_committed_event(event)?,
        )),
    })
}

/// gRPC transport server bound to a durable host runtime.
#[derive(Clone)]
pub struct GrpcTransport {
    shared: Arc<GrpcShared>,
}

impl GrpcTransport {
    /// `heartbeat_interval` must be the SAME configured worker cadence the
    /// pool beats at (`worker.heartbeat_interval`) — see ADR-0003 I4. The
    /// detached Confirm-tier drive spawned from `DecideConfirmation` beats at
    /// it, and the subagent stall floor is derived from it.
    #[must_use]
    pub fn new(
        stores: StoreRegistry,
        runtime: Arc<ExecutionRuntime>,
        health: Arc<HealthSurface>,
        shutdown: CancellationToken,
        lease_duration: time::Duration,
        heartbeat_interval: std::time::Duration,
    ) -> Self {
        Self::with_admission(
            stores,
            runtime,
            health,
            shutdown,
            lease_duration,
            heartbeat_interval,
            AdmissionConfig::default(),
        )
    }

    /// Like [`Self::new`] but with explicit admission back-pressure and
    /// input-size limits (Phase 10 · E).
    #[must_use]
    pub fn with_admission(
        stores: StoreRegistry,
        runtime: Arc<ExecutionRuntime>,
        health: Arc<HealthSurface>,
        shutdown: CancellationToken,
        lease_duration: time::Duration,
        heartbeat_interval: std::time::Duration,
        admission: AdmissionConfig,
    ) -> Self {
        Self {
            shared: Arc::new(GrpcShared::new(
                stores,
                runtime,
                health,
                shutdown,
                lease_duration,
                heartbeat_interval,
                admission,
            )),
        }
    }

    /// # Errors
    ///
    /// Returns an error if binding the TCP listener fails or if the gRPC server
    /// exits with a transport error.
    pub async fn serve(self, addr: SocketAddr) -> Result<()> {
        let listener = TcpListener::bind(addr)
            .await
            .with_context(|| format!("binding gRPC listener on {addr}"))?;
        self.serve_listener(listener).await
    }

    /// # Errors
    ///
    /// Returns an error if the health loop or gRPC transport cannot be served on
    /// the provided listener.
    pub async fn serve_listener(self, listener: TcpListener) -> Result<()> {
        let (reporter, health_service) = tonic_health::server::health_reporter();
        let control_service = GrpcControlService {
            shared: Arc::clone(&self.shared),
        };
        let event_service = GrpcEventService {
            shared: Arc::clone(&self.shared),
        };

        let shutdown = self.shared.shutdown.clone();
        let health = Arc::clone(&self.shared.health);
        let health_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(200));
            loop {
                let ready = health.snapshot().is_ready();
                if ready {
                    reporter
                        .set_serving::<AgentControlServiceServer<GrpcControlService>>()
                        .await;
                    reporter
                        .set_serving::<AgentEventServiceServer<GrpcEventService>>()
                        .await;
                } else {
                    reporter
                        .set_not_serving::<AgentControlServiceServer<GrpcControlService>>()
                        .await;
                    reporter
                        .set_not_serving::<AgentEventServiceServer<GrpcEventService>>()
                        .await;
                }

                tokio::select! {
                    () = shutdown.cancelled() => break,
                    _ = interval.tick() => {}
                }
            }
        });

        // Stamp `rpc.server.duration` on every inbound call when
        // the `otel` feature is on. The cfg
        // gate keeps the default build dependency-free; the layer
        // call is identical in either case.
        #[cfg(feature = "otel")]
        let mut server = Server::builder()
            .http2_keepalive_interval(Some(std::time::Duration::from_secs(20)))
            .http2_keepalive_timeout(Some(std::time::Duration::from_secs(10)))
            .tcp_keepalive(Some(std::time::Duration::from_mins(1)))
            .layer(crate::observability::grpc_layer::TelemetryLayer::new());
        #[cfg(not(feature = "otel"))]
        let mut server = Server::builder()
            .http2_keepalive_interval(Some(std::time::Duration::from_secs(20)))
            .http2_keepalive_timeout(Some(std::time::Duration::from_secs(10)))
            .tcp_keepalive(Some(std::time::Duration::from_mins(1)));

        // Bound the transport-level decode buffer so an oversized frame
        // is rejected before it is fully decoded into memory (Phase 10 ·
        // E). The application-level input-size check returns a cleaner
        // INVALID_ARGUMENT for normal payloads; this is the hard ceiling.
        let control_server = AgentControlServiceServer::new(control_service)
            .max_decoding_message_size(self.shared.admission.max_decoding_message_bytes);

        server
            .add_service(health_service)
            .add_service(control_server)
            .add_service(AgentEventServiceServer::new(event_service))
            .serve_with_incoming_shutdown(
                TcpListenerStream::new(listener),
                self.shared.shutdown.cancelled(),
            )
            .await
            .context("serving gRPC transport")?;

        health_task.abort();
        let _ = health_task.await;
        Ok(())
    }
}

/// Process-local daemon that runs the host workers and gRPC listener together.
pub struct LocalDaemon {
    addr: SocketAddr,
    shutdown: CancellationToken,
    host_task: JoinHandle<Result<()>>,
    grpc_task: JoinHandle<Result<()>>,
}

impl LocalDaemon {
    /// # Errors
    ///
    /// Returns an error if the durable stores, host runtime, or local gRPC
    /// listener cannot be initialized.
    pub async fn start(
        config: ServiceConfig,
        definition_registry: Arc<dyn AgentDefinitionRegistry>,
        runtime: Arc<ExecutionRuntime>,
    ) -> Result<Self> {
        Self::start_with_store_decorator(config, definition_registry, runtime, |stores| stores)
            .await
    }

    /// Start the daemon like [`Self::start`], but let the caller wrap the
    /// assembled [`StoreRegistry`] before the daemon uses it.
    ///
    /// Embedding hosts use this to decorate individual store handles —
    /// e.g. intercepting [`agent_server::journal::store::AgentTaskStore::complete_task_with_result`]
    /// on `task_store` to post-process subagent results — without
    /// mirroring the whole daemon assembly.
    ///
    /// # Contract
    ///
    /// - The registry handed to `decorate` is exactly the one
    ///   [`StoreRegistry::from_config`] builds for `config.storage`,
    ///   with all of its constructor-established wiring intact: the
    ///   backend handle that [`StoreRegistry::initialize`] uses to run
    ///   migrations, the process-wide `event_notifier`, and
    ///   `confirm_drive_cancels`.
    /// - The daemon uses the **returned** registry everywhere: store
    ///   initialization, the [`ServiceHost`] worker pool and sweep
    ///   tasks, and the gRPC transport. There is exactly one registry;
    ///   both planes share the same decorated handles.
    /// - Decorators should only replace the `pub` store fields they
    ///   mean to intercept (e.g. `task_store`) by wrapping the existing
    ///   handle, and must delegate every trait method they do not
    ///   intercept to the wrapped store — the daemon's CAS/lease
    ///   discipline runs through those methods.
    /// - Decorators must not swap `event_notifier` or
    ///   `confirm_drive_cancels` for fresh instances and must not
    ///   substitute a registry assembled from a different backend:
    ///   return the registry that was passed in, with fields wrapped.
    ///   Both handles are checked with debug assertions after
    ///   `decorate` returns.
    /// - `decorate` runs before [`StoreRegistry::initialize`], so
    ///   decorators must not perform store I/O inside `decorate`
    ///   itself — on Postgres the schema has not been migrated yet at
    ///   that point.
    /// - On the in-memory backend, `cancel_tree`'s terminal `Cancelled`
    ///   markers are committed through sink handles bound at registry
    ///   construction (before decoration), so an `event_repo` decorator
    ///   observes every event *except* those markers — the same class
    ///   as the durable backends, whose atomic commits bypass
    ///   `event_repo` decoration entirely.
    ///
    /// # Errors
    ///
    /// Returns an error if the durable stores, host runtime, or local gRPC
    /// listener cannot be initialized.
    pub async fn start_with_store_decorator(
        config: ServiceConfig,
        definition_registry: Arc<dyn AgentDefinitionRegistry>,
        runtime: Arc<ExecutionRuntime>,
        decorate: impl FnOnce(StoreRegistry) -> StoreRegistry + Send,
    ) -> Result<Self> {
        let stores = StoreRegistry::from_config(&config.storage, definition_registry)
            .context("creating daemon stores")?;
        let event_notifier = Arc::clone(&stores.event_notifier);
        let confirm_drive_cancels = Arc::clone(&stores.confirm_drive_cancels);
        let stores = decorate(stores);
        // Two pointer comparisons catch both a forbidden wiring swap
        // and a whole-registry substitution (e.g. returning a fresh
        // `StoreRegistry::in_memory`). A violation is a decorator
        // contract bug, not runtime input — hence debug assertions.
        debug_assert!(
            Arc::ptr_eq(&event_notifier, &stores.event_notifier),
            "store decorator must not replace the registry's event_notifier",
        );
        debug_assert!(
            Arc::ptr_eq(&confirm_drive_cancels, &stores.confirm_drive_cancels),
            "store decorator must not replace the registry's confirm_drive_cancels",
        );
        stores
            .initialize()
            .await
            .context("initializing daemon stores")?;
        let host = ServiceHost::with_stores(config.clone(), stores.clone(), Arc::clone(&runtime))
            .context("creating daemon host")?;
        let health = Arc::clone(host.health());
        let shutdown = host.shutdown_token();
        let grpc = GrpcTransport::with_admission(
            stores,
            runtime,
            health,
            shutdown.clone(),
            config.worker.lease_duration(),
            config.worker.heartbeat_interval(),
            config.admission.clone(),
        );

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .context("binding local daemon listener")?;
        let addr = listener
            .local_addr()
            .context("reading local daemon listener address")?;

        let host_task = tokio::spawn(async move { host.run().await });
        let grpc_task = tokio::spawn(async move { grpc.serve_listener(listener).await });

        info!(%addr, "local daemon started");
        Ok(Self {
            addr,
            shutdown,
            host_task,
            grpc_task,
        })
    }

    #[must_use]
    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    #[must_use]
    pub fn endpoint(&self) -> String {
        format!("http://{}", self.addr)
    }

    #[must_use]
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// # Errors
    ///
    /// Returns an error if either the host task or gRPC task fails while the
    /// daemon is shutting down.
    pub async fn stop(self) -> Result<()> {
        self.shutdown.cancel();
        self.grpc_task.await.context("joining gRPC task")??;
        self.host_task.await.context("joining host task")??;
        Ok(())
    }
}

/// Caller-minted thread-creation claims are stored in the shared
/// `request_id` keyspace under this prefix (see
/// [`agent_server::journal::thread_store::creation_identity_key`]).
/// Rejecting the prefix at the RPC boundary makes a caller/claim
/// collision unrepresentable rather than merely unlikely.
const RESERVED_REQUEST_ID_PREFIX: &str = "agent-sdk:";

fn require_request_id(request_id: &str) -> RpcResult<()> {
    let trimmed = request_id.trim();
    if trimmed.is_empty() {
        return Err(Status::invalid_argument("request_id is required").into());
    }
    if trimmed.starts_with(RESERVED_REQUEST_ID_PREFIX) {
        return Err(Status::invalid_argument(
            "the 'agent-sdk:' request_id prefix is reserved for server-internal claims",
        )
        .into());
    }
    Ok(())
}

fn parse_thread_id(thread_id: &str) -> RpcResult<ThreadId> {
    if thread_id.trim().is_empty() {
        return Err(Status::invalid_argument("thread_id is required").into());
    }
    Ok(ThreadId::from_string(thread_id))
}

/// Parse a caller-supplied destination thread id. Empty means
/// server-minted; a non-empty id must be a UUID so arbitrary caller
/// strings never become durable primary keys.
fn parse_caller_thread_id(thread_id: &str) -> RpcResult<Option<ThreadId>> {
    let trimmed = thread_id.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if uuid::Uuid::parse_str(trimmed).is_err() {
        return Err(
            Status::invalid_argument(format!("thread_id must be a UUID, got {trimmed:?}")).into(),
        );
    }
    Ok(Some(ThreadId::from_string(trimmed)))
}

fn fork_destination(
    caller_thread_id: Option<ThreadId>,
    source_thread_id: &ThreadId,
    fork_after_committed_turns: u32,
) -> (ThreadId, Option<ThreadCreation>) {
    let Some(thread_id) = caller_thread_id else {
        return (ThreadId::new(), None);
    };
    (
        thread_id,
        Some(ThreadCreation::Fork {
            source_thread_id: source_thread_id.clone(),
            fork_after_committed_turns,
        }),
    )
}

fn parse_task_id(task_id: &str) -> RpcResult<AgentTaskId> {
    if task_id.trim().is_empty() {
        return Err(Status::invalid_argument("task_id is required").into());
    }
    Ok(AgentTaskId::from_string(task_id))
}

/// Log-line budget for the client-controlled `CancelTask.reason` string.
/// Matches the spirit of `SubmitThreadWork`'s input caps: the field is
/// documented as logs-only, so bounding it here is the whole guard.
const MAX_CANCEL_REASON_LOG_CHARS: usize = 256;

/// Bound a client-controlled string before it reaches structured logs,
/// truncating on a char boundary and marking the cut with an ellipsis.
/// Borrowed (no allocation) when the value is already within budget.
fn bounded_log_str(value: &str, max_chars: usize) -> std::borrow::Cow<'_, str> {
    match value.char_indices().nth(max_chars) {
        None => std::borrow::Cow::Borrowed(value),
        Some((byte_index, _)) => std::borrow::Cow::Owned(format!("{}…", &value[..byte_index])),
    }
}

fn internal_status(
    context: &'static str,
) -> impl Fn(anyhow::Error) -> Status + Copy + Send + Sync + 'static {
    move |error| Status::internal(format!("{context}: {error:#}"))
}

fn map_thread_creation_error(
    context: &'static str,
) -> impl Fn(anyhow::Error) -> Status + Copy + Send + Sync + 'static {
    move |error| {
        if let Some(conflict) = error.downcast_ref::<ThreadIdConflict>() {
            return Status::already_exists(conflict.to_string());
        }
        internal_status(context)(error)
    }
}

fn as_stream_status(
    context: &'static str,
) -> impl Fn(anyhow::Error) -> Status + Copy + Send + Sync + 'static {
    internal_status(context)
}

fn not_found_status(resource_kind: &str, resource_id: &str) -> Status {
    Status::not_found(format!("{resource_kind} '{resource_id}' was not found"))
}

fn idempotency_conflict_status(method: &str, request_id: &str) -> Status {
    Status::already_exists(format!(
        "request_id '{request_id}' was already used for a different {method} request"
    ))
}

/// Map a transactional [`SubmitRootTurnError`] onto the gRPC status the
/// caller should see.
fn map_submit_root_error(method: &str, request_id: &str, error: SubmitRootTurnError) -> Status {
    match error {
        SubmitRootTurnError::IdempotencyConflict => idempotency_conflict_status(method, request_id),
        SubmitRootTurnError::QueueDepthExceeded { cap, current_depth } => {
            Status::resource_exhausted(format!(
                "thread queued-root depth {current_depth} reached the configured cap {cap}; \
                 retry after in-flight roots drain"
            ))
        }
        SubmitRootTurnError::Other(error) => {
            Status::internal(format!("submitting root turn: {error:#}"))
        }
    }
}

fn submit_work_fingerprint(request: &pb::SubmitThreadWorkRequest) -> Vec<u8> {
    let mut request = request.clone();
    request.request_id.clear();
    request.encode_to_vec()
}

/// Decode the durable references stored in an idempotency record's
/// `result_json` into the operation-specific result struct.
fn decode_idempotency_result<T: serde::de::DeserializeOwned>(
    result_json: &serde_json::Value,
) -> Result<T, Status> {
    serde_json::from_value(result_json.clone())
        .map_err(|error| Status::internal(format!("decoding stored idempotency result: {error}")))
}

/// Return the cumulative `total_turns` count carried by a `Done`
/// event, or `None` for any other event variant.
///
/// Used by the fork-state copy to find the cutoff sequence: events
/// with no turn carry-over (text deltas, tool calls, `Start`,
/// `TurnComplete`) take the running turn from the most recent
/// `Done`, so consecutive sub-`Done` events at sequence S inherit
/// the same boundary as the next `Done`. A `fork_after` cap maps to
/// "every event whose enclosing turn `<= fork_after`": we walk
/// events in sequence, take while the *running* turn `<= cap`, and
/// stop the moment a `Done` increments the running turn past it.
///
/// Returning `None` for non-boundary events lets the caller's
/// `take_while` accept them under the current running turn rather
/// than reset it — the agent-server's commit pipeline always emits
/// the terminal marker last, so any non-boundary event seen at
/// sequence S belongs to a turn whose marker is at some sequence
/// `>= S`.
///
/// `Done` is not the only terminal marker the pipeline commits:
/// `BudgetExceeded` is emitted in place of `Done` when a budget trips
/// (carrying `total_turns`), and `Cancelled` is the durable marker an
/// effective `CancelTask` commits (carrying `turn`). All three are
/// boundaries — without them a fork below a budget-stop or cancel
/// point would copy the terminal marker past the cutoff and the fork
/// would immediately read as completed/cancelled.
///
/// Slot-shifted turns: a successor turn that lost its bootstrapped
/// slot to a cancelled predecessor's salvage
/// commits with `Done.total_turns` remapped to the landed slot, so
/// this boundary walk stays consistent with the thread's committed
/// turn count and the checkpoint numbering. Only the already-committed
/// `Start` may carry the stale index — deliberately not a boundary
/// here (see the `StartEvent` contract in events.proto: boundaries
/// derive from event order, completion-side indices are
/// authoritative).
const fn turn_number_for_event(event: &agent_sdk_foundation::events::AgentEvent) -> Option<u64> {
    match event {
        agent_sdk_foundation::events::AgentEvent::Done { total_turns, .. }
        | agent_sdk_foundation::events::AgentEvent::BudgetExceeded { total_turns, .. } => {
            Some(*total_turns as u64)
        }
        agent_sdk_foundation::events::AgentEvent::Cancelled { turn, .. } => Some(*turn as u64),
        _ => None,
    }
}

/// Source events whose enclosing turn is `<= fork_after`, in commit order.
///
/// Walks the committed log and takes events while the running turn (per
/// [`turn_number_for_event`]) stays within the cap — stopping at the first
/// boundary marker (`Done` / `BudgetExceeded` / `Cancelled`) past it, so a
/// fork below a terminal marker never inherits it.
fn events_up_to_fork_boundary(
    source_events: Vec<agent_server::CommittedEvent>,
    fork_after: u32,
) -> Vec<agent_sdk_foundation::events::AgentEvent> {
    source_events
        .into_iter()
        .take_while(|committed| {
            turn_number_for_event(&committed.event).is_none_or(|turn| turn <= u64::from(fork_after))
        })
        .map(|committed| committed.event)
        .collect()
}

/// Rewrite the `thread_id` field on a JSON `AgentState` snapshot.
///
/// The forked thread inherits the source's checkpointed agent state
/// (`turn_count`, `total_usage`, `metadata`, `created_at`) but lives under a
/// new thread id. The agent loop's [`StagedStateStore::save`] guard
/// rejects state writes whose `thread_id` doesn't match the bound
/// thread, so the snapshot's `thread_id` field has to be rewritten in
/// place before it lands on the fork's checkpoint. Operating directly
/// on the JSON keeps the rewrite forward-compatible with new
/// `AgentState` fields — we only touch the one key we know about.
///
/// `Value::Null` snapshots (fresh-thread case) round-trip unchanged,
/// matching the contract `from_recovery_view` already honours when
/// the snapshot is null.
/// Pluck the cumulative `total_usage` field out of an
/// `agent_state_snapshot` JSON value.
///
/// The snapshot is whatever
/// [`agent_sdk_foundation::AgentState`] serializes to — the relevant
/// shape is `{ "thread_id": ..., "turn_count": ...,
/// "total_usage": { "input_tokens": N, "output_tokens": M, … },
/// "metadata": ..., "created_at": ... }`. The fork uses this
/// value to seed the destination thread aggregate's `total_usage`
/// so cost reporting picks up where the source left off rather
/// than zeroing out.
///
/// Failure modes:
/// - `Value::Null` (fresh-thread case): returns
///   [`TokenUsage::default()`].
/// - Missing `total_usage` key: returns
///   [`TokenUsage::default()`] (forward-compat with snapshots
///   from older `AgentState` shapes).
/// - Present-but-malformed `total_usage`: errors. A snapshot we
///   can't parse is corruption; better to surface it than to
///   silently land a fork with the wrong total.
fn total_usage_from_state_snapshot(
    snapshot: &serde_json::Value,
) -> anyhow::Result<agent_sdk_foundation::TokenUsage> {
    use anyhow::Context as _;
    if snapshot.is_null() {
        return Ok(agent_sdk_foundation::TokenUsage::default());
    }
    let object = snapshot
        .as_object()
        .context("agent_state_snapshot is not a JSON object")?;
    let Some(value) = object.get("total_usage") else {
        return Ok(agent_sdk_foundation::TokenUsage::default());
    };
    serde_json::from_value(value.clone()).context("agent_state_snapshot.total_usage is malformed")
}

fn rewrite_state_snapshot_thread_id(
    snapshot: &serde_json::Value,
    new_thread_id: &ThreadId,
) -> anyhow::Result<serde_json::Value> {
    use anyhow::Context as _;
    if snapshot.is_null() {
        return Ok(serde_json::Value::Null);
    }
    let mut snapshot = snapshot.clone();
    let object = snapshot
        .as_object_mut()
        .context("agent_state_snapshot is not a JSON object")?;
    object.insert(
        "thread_id".to_string(),
        serde_json::Value::String(new_thread_id.0.clone()),
    );
    Ok(snapshot)
}

fn decide_confirmation_fingerprint(request: &pb::DecideConfirmationRequest) -> Vec<u8> {
    let mut request = request.clone();
    request.request_id.clear();
    request.encode_to_vec()
}

/// Fingerprint for the REQUEST-level idempotency row (keyed by the
/// caller's `request_id`): the whole wire request minus the id, so a
/// reused `request_id` with any changed parameter — including a
/// different caller-minted destination — conflicts. Distinct from
/// [`ThreadCreation::fingerprint`], which keys the CREATION claim under
/// the destination thread's identity and deliberately excludes
/// transport request ids.
fn fork_thread_fingerprint(request: &pb::ForkThreadRequest) -> Vec<u8> {
    let mut request = request.clone();
    request.request_id.clear();
    request.encode_to_vec()
}

const fn map_thread_status(status: agent_server::ThreadStatus) -> i32 {
    match status {
        agent_server::ThreadStatus::Active => pb::ThreadStatus::Active as i32,
        agent_server::ThreadStatus::Completed => pb::ThreadStatus::Completed as i32,
    }
}

const fn map_task_kind(kind: JournalTaskKind) -> i32 {
    match kind {
        JournalTaskKind::RootTurn => pb::TaskKind::RootTurn as i32,
        JournalTaskKind::ToolRuntime => pb::TaskKind::ToolRuntime as i32,
        JournalTaskKind::Subagent => pb::TaskKind::Subagent as i32,
    }
}

const fn map_task_status(status: JournalTaskStatus) -> i32 {
    match status {
        JournalTaskStatus::Queued => pb::TaskStatus::Queued as i32,
        JournalTaskStatus::Pending => pb::TaskStatus::Pending as i32,
        JournalTaskStatus::Running => pb::TaskStatus::Running as i32,
        JournalTaskStatus::WaitingOnChildren => pb::TaskStatus::WaitingOnChildren as i32,
        JournalTaskStatus::AwaitingConfirmation => pb::TaskStatus::AwaitingConfirmation as i32,
        JournalTaskStatus::Completed => pb::TaskStatus::Completed as i32,
        JournalTaskStatus::Failed => pb::TaskStatus::Failed as i32,
        JournalTaskStatus::Cancelled => pb::TaskStatus::Cancelled as i32,
    }
}

const fn map_tool_tier(tier: ToolTier) -> i32 {
    match tier {
        ToolTier::Observe => pb::ToolTier::Observe as i32,
        ToolTier::Confirm => pb::ToolTier::Confirm as i32,
    }
}

const fn map_message_role(role: llm::Role) -> i32 {
    match role {
        llm::Role::User => pb::MessageRole::User as i32,
        llm::Role::Assistant => pb::MessageRole::Assistant as i32,
    }
}

fn map_token_usage(usage: &TokenUsage) -> pb::TokenUsage {
    pb::TokenUsage {
        input_tokens: u64::from(usage.input_tokens),
        output_tokens: u64::from(usage.output_tokens),
        cached_input_tokens: u64::from(usage.cached_input_tokens),
        cache_creation_input_tokens: u64::from(usage.cache_creation_input_tokens),
    }
}

fn map_timestamp(timestamp: OffsetDateTime) -> RpcResult<ProtoTimestamp> {
    Ok(ProtoTimestamp {
        seconds: timestamp.unix_timestamp(),
        nanos: i32::try_from(timestamp.nanosecond())
            .map_err(|_| Status::internal("timestamp nanoseconds out of range"))?,
    })
}

fn map_optional_timestamp(timestamp: Option<OffsetDateTime>) -> RpcResult<Option<ProtoTimestamp>> {
    timestamp.map(map_timestamp).transpose()
}

fn map_duration(duration: std::time::Duration) -> RpcResult<ProtoDuration> {
    Ok(ProtoDuration {
        seconds: i64::try_from(duration.as_secs())
            .map_err(|_| Status::internal("duration seconds out of range"))?,
        nanos: i32::try_from(duration.subsec_nanos())
            .map_err(|_| Status::internal("duration nanoseconds out of range"))?,
    })
}

fn map_json_value(value: &serde_json::Value) -> RpcResult<ProtoValue> {
    let kind = match value {
        serde_json::Value::Null => ProtoValueKind::NullValue(0),
        serde_json::Value::Bool(value) => ProtoValueKind::BoolValue(*value),
        serde_json::Value::Number(number) => ProtoValueKind::NumberValue(
            number
                .as_f64()
                .ok_or_else(|| Status::internal("json number could not be represented as f64"))?,
        ),
        serde_json::Value::String(value) => ProtoValueKind::StringValue(value.clone()),
        serde_json::Value::Array(values) => ProtoValueKind::ListValue(ProtoListValue {
            values: values
                .iter()
                .map(map_json_value)
                .collect::<RpcResult<Vec<_>>>()?,
        }),
        serde_json::Value::Object(values) => ProtoValueKind::StructValue(ProtoStruct {
            fields: values
                .iter()
                .map(|(key, value)| Ok((key.clone(), map_json_value(value)?)))
                .collect::<RpcResult<BTreeMap<_, _>>>()?,
        }),
    };
    Ok(ProtoValue { kind: Some(kind) })
}

fn map_binary_attachment(source: &ContentSource) -> RpcResult<pb::BinaryAttachment> {
    Ok(pb::BinaryAttachment {
        media_type: source.media_type.clone(),
        data: base64::engine::general_purpose::STANDARD
            .decode(&source.data)
            .map_err(|error| {
                Status::internal(format!("invalid base64 attachment payload: {error}"))
            })?,
    })
}

fn map_tool_result(result: &ToolResult) -> RpcResult<pb::ToolResult> {
    Ok(pb::ToolResult {
        success: result.success,
        output: result.output.clone(),
        data: result.data.as_ref().map(map_json_value).transpose()?,
        documents: result
            .documents
            .iter()
            .map(map_binary_attachment)
            .collect::<RpcResult<Vec<_>>>()?,
        duration_ms: result.duration_ms,
    })
}

fn map_prepared_operation(
    operation: &agent_sdk_foundation::ListenExecutionContext,
) -> RpcResult<pb::PreparedOperation> {
    Ok(pb::PreparedOperation {
        operation_id: operation.operation_id.clone(),
        revision: operation.revision,
        snapshot: Some(map_json_value(&operation.snapshot)?),
        expires_at: map_optional_timestamp(operation.expires_at)?,
    })
}

fn map_message(message: &llm::Message) -> RpcResult<pb::ConversationMessage> {
    let content = match &message.content {
        llm::Content::Text(text) => Some(pb::conversation_message::Content::Text(text.clone())),
        llm::Content::Blocks(blocks) => Some(pb::conversation_message::Content::Blocks(
            pb::ConversationContentList {
                items: blocks
                    .iter()
                    // Provider-owned reasoning state is required for internal
                    // replay but must never be projected through the public
                    // conversation API.
                    .filter(|block| !matches!(block, ContentBlock::OpaqueReasoning { .. }))
                    .map(map_content_block)
                    .collect::<RpcResult<Vec<_>>>()?,
            },
        )),
    };

    Ok(pb::ConversationMessage {
        role: map_message_role(message.role),
        content,
    })
}

fn map_content_block(block: &ContentBlock) -> RpcResult<pb::ConversationContentBlock> {
    let block = match block {
        ContentBlock::Text { text } => {
            pb::conversation_content_block::Block::Text(pb::TextBlock { text: text.clone() })
        }
        ContentBlock::Thinking {
            thinking,
            signature,
        } => pb::conversation_content_block::Block::Thinking(pb::ThinkingBlock {
            thinking: thinking.clone(),
            signature: signature.clone(),
        }),
        ContentBlock::RedactedThinking { data } => {
            pb::conversation_content_block::Block::RedactedThinking(pb::RedactedThinkingBlock {
                data: data.clone(),
            })
        }
        ContentBlock::ToolUse {
            id,
            name,
            input,
            thought_signature,
        } => pb::conversation_content_block::Block::ToolUse(pb::ToolUseBlock {
            tool_call_id: id.clone(),
            name: name.clone(),
            input: Some(map_json_value(input)?),
            thought_signature: thought_signature.clone(),
        }),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => pb::conversation_content_block::Block::ToolResult(pb::ToolResultBlock {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
        }),
        ContentBlock::Image { source } => {
            pb::conversation_content_block::Block::Image(map_binary_attachment(source)?)
        }
        ContentBlock::Document { source } => {
            pb::conversation_content_block::Block::Document(map_binary_attachment(source)?)
        }
        // `ContentBlock` is `#[non_exhaustive]`; a block kind this service
        // version cannot project onto the wire protocol is a hard error
        // rather than a silently dropped block.
        _ => {
            return Err(Status::internal(
                "unrecognized content block kind in conversation message",
            )
            .into());
        }
    };

    Ok(pb::ConversationContentBlock { block: Some(block) })
}

fn map_submitted_input_item(item: &pb::UserInputItem) -> RpcResult<SubmittedInputItem> {
    match item.item.as_ref() {
        Some(pb::user_input_item::Item::Text(text)) => {
            Ok(SubmittedInputItem::Text { text: text.clone() })
        }
        Some(pb::user_input_item::Item::Image(image)) => Ok(SubmittedInputItem::Image {
            media_type: image.media_type.clone(),
            data_base64: base64::engine::general_purpose::STANDARD.encode(&image.data),
        }),
        Some(pb::user_input_item::Item::Document(document)) => Ok(SubmittedInputItem::Document {
            media_type: document.media_type.clone(),
            data_base64: base64::engine::general_purpose::STANDARD.encode(&document.data),
        }),
        None => Err(Status::invalid_argument("user input item is required").into()),
    }
}

fn map_confirmation_decision(
    decision: &pb::ConfirmationDecision,
) -> RpcResult<ConfirmationDecision> {
    match decision.decision.as_ref() {
        Some(pb::confirmation_decision::Decision::Approved(_)) => {
            Ok(ConfirmationDecision::Approved)
        }
        Some(pb::confirmation_decision::Decision::Rejected(rejected)) => {
            if rejected.reason.trim().is_empty() {
                return Err(
                    Status::invalid_argument("rejected confirmations require a reason").into(),
                );
            }
            Ok(ConfirmationDecision::Rejected {
                reason: rejected.reason.clone(),
            })
        }
        Some(pb::confirmation_decision::Decision::TimedOut(_)) => Ok(ConfirmationDecision::Timeout),
        None => Err(Status::invalid_argument("confirmation decision is required").into()),
    }
}

fn map_follow_mode(follow_mode: i32) -> RpcResult<pb::FollowMode> {
    match pb::FollowMode::try_from(follow_mode) {
        Ok(pb::FollowMode::ReplayOnly) => Ok(pb::FollowMode::ReplayOnly),
        Ok(pb::FollowMode::ReplayAndFollow) => Ok(pb::FollowMode::ReplayAndFollow),
        _ => Err(
            Status::invalid_argument("follow_mode must be replay_only or replay_and_follow").into(),
        ),
    }
}

fn map_committed_event(event: &agent_server::CommittedEvent) -> RpcResult<pb::EventEnvelope> {
    let payload = map_event_payload(&event.event)?;
    Ok(pb::EventEnvelope {
        event_id: event.event_id.to_string(),
        thread_id: event.thread_id.0.clone(),
        sequence: event.sequence,
        commit_time: Some(map_timestamp(event.timestamp)?),
        event: Some(payload),
    })
}

fn map_event_payload(event: &AgentEvent) -> RpcResult<pb::event_envelope::Event> {
    if let Some(payload) = map_message_event_payload(event)? {
        return Ok(payload);
    }
    if let Some(payload) = map_tool_event_payload(event)? {
        return Ok(payload);
    }
    if let Some(payload) = map_subagent_event_payload(event) {
        return Ok(payload);
    }
    map_lifecycle_event_payload(event)
}

fn map_subagent_event_payload(event: &AgentEvent) -> Option<pb::event_envelope::Event> {
    let AgentEvent::SubagentProgress {
        subagent_id,
        subagent_name,
        nickname,
        child_thread_id,
        child_root_task_id,
        subagent_task_id,
        max_turns,
        current_turn,
        model,
        tool_name,
        tool_context,
        completed,
        success,
        tool_count,
        total_tokens,
    } = event
    else {
        return None;
    };
    Some(pb::event_envelope::Event::SubagentProgress(
        pb::SubagentProgressEvent {
            subagent_id: subagent_id.clone(),
            subagent_name: subagent_name.clone(),
            nickname: nickname.clone(),
            child_thread_id: child_thread_id.as_ref().map(ToString::to_string),
            child_root_task_id: child_root_task_id.clone(),
            subagent_task_id: subagent_task_id.clone(),
            max_turns: *max_turns,
            current_turn: *current_turn,
            model: model.clone(),
            tool_name: tool_name.clone(),
            tool_context: tool_context.clone(),
            completed: *completed,
            success: *success,
            tool_count: *tool_count,
            total_tokens: *total_tokens,
        },
    ))
}

fn map_message_event_payload(event: &AgentEvent) -> RpcResult<Option<pb::event_envelope::Event>> {
    match event {
        AgentEvent::Start {
            thread_id,
            turn,
            emitter_task_id,
        } => Ok(Some(pb::event_envelope::Event::Start(pb::StartEvent {
            thread_id: thread_id.0.clone(),
            turn: map_u64(*turn, "turn")?,
            emitter_task_id: emitter_task_id.clone(),
        }))),
        AgentEvent::Thinking { message_id, text } => Ok(Some(pb::event_envelope::Event::Thinking(
            pb::ThinkingEvent {
                message_id: message_id.clone(),
                text: text.clone(),
            },
        ))),
        AgentEvent::ThinkingDelta { message_id, delta } => Ok(Some(
            pb::event_envelope::Event::ThinkingDelta(pb::ThinkingDeltaEvent {
                message_id: message_id.clone(),
                delta: delta.clone(),
            }),
        )),
        AgentEvent::TextDelta { message_id, delta } => Ok(Some(
            pb::event_envelope::Event::TextDelta(pb::TextDeltaEvent {
                message_id: message_id.clone(),
                delta: delta.clone(),
            }),
        )),
        AgentEvent::Text { message_id, text } => {
            Ok(Some(pb::event_envelope::Event::Text(pb::TextEvent {
                message_id: message_id.clone(),
                text: text.clone(),
            })))
        }
        AgentEvent::Refusal { message_id, text } => {
            Ok(Some(pb::event_envelope::Event::Refusal(pb::RefusalEvent {
                message_id: message_id.clone(),
                text: text.clone(),
            })))
        }
        AgentEvent::UserInput { thread_id, content } => {
            let content_proto: Vec<pb::ConversationContentBlock> = content
                .iter()
                .map(map_content_block)
                .collect::<RpcResult<_>>()?;
            Ok(Some(pb::event_envelope::Event::UserInput(
                pb::UserInputEvent {
                    thread_id: thread_id.0.clone(),
                    content: content_proto,
                },
            )))
        }
        _ => Ok(None),
    }
}

fn map_tool_event_payload(event: &AgentEvent) -> RpcResult<Option<pb::event_envelope::Event>> {
    match event {
        AgentEvent::ToolCallStart {
            id,
            name,
            display_name,
            input,
            tier,
        } => Ok(Some(pb::event_envelope::Event::ToolCallStart(
            pb::ToolCallStartEvent {
                tool_call_id: id.clone(),
                name: name.clone(),
                display_name: display_name.clone(),
                input: Some(map_json_value(input)?),
                tier: map_tool_tier(*tier),
            },
        ))),
        AgentEvent::ToolCallEnd {
            id,
            name,
            display_name,
            result,
        } => Ok(Some(pb::event_envelope::Event::ToolCallEnd(
            pb::ToolCallEndEvent {
                tool_call_id: id.clone(),
                name: name.clone(),
                display_name: display_name.clone(),
                result: Some(map_tool_result(result)?),
            },
        ))),
        AgentEvent::ToolProgress {
            id,
            name,
            display_name,
            stage,
            message,
            data,
        } => Ok(Some(pb::event_envelope::Event::ToolProgress(
            pb::ToolProgressEvent {
                tool_call_id: id.clone(),
                name: name.clone(),
                display_name: display_name.clone(),
                stage: stage.clone(),
                message: message.clone(),
                data: data.as_ref().map(map_json_value).transpose()?,
            },
        ))),
        AgentEvent::ToolRequiresConfirmation {
            id,
            name,
            display_name,
            input,
            description,
        } => Ok(Some(pb::event_envelope::Event::ToolRequiresConfirmation(
            pb::ToolRequiresConfirmationEvent {
                tool_call_id: id.clone(),
                name: name.clone(),
                display_name: display_name.clone(),
                input: Some(map_json_value(input)?),
                description: description.clone(),
            },
        ))),
        _ => Ok(None),
    }
}

/// Project a terminal run-completion event onto the wire `DoneEvent`.
///
/// Shared by [`AgentEvent::Done`] and [`AgentEvent::BudgetExceeded`]: the
/// proto has no dedicated budget event, so a budget-terminated run reuses
/// the `done` frame with `stop_reason` distinguishing the disposition.
/// This keeps the streaming contract additive — a replay/follow consumer
/// always receives a closing `done` frame and the stream terminates (see
/// [`terminal_close_reason`]) instead of failing with INTERNAL on an unmapped
/// variant. Both variants carry the run's wall-clock duration.
fn map_done_event(
    thread_id: &ThreadId,
    total_turns: usize,
    total_usage: &TokenUsage,
    duration: std::time::Duration,
    stop_reason: pb::RunStopReason,
    estimated_cost_usd: Option<f64>,
    emitter_task_id: Option<String>,
) -> RpcResult<pb::event_envelope::Event> {
    Ok(pb::event_envelope::Event::Done(pb::DoneEvent {
        thread_id: thread_id.0.clone(),
        total_turns: map_u64(total_turns, "total_turns")?,
        total_usage: Some(map_token_usage(total_usage)),
        duration: Some(map_duration(duration)?),
        stop_reason: stop_reason.into(),
        estimated_cost_usd,
        emitter_task_id,
    }))
}

/// Wire disposition for a budget-terminated run, by the limit that tripped.
const fn budget_stop_reason(
    limit: agent_sdk_foundation::types::BudgetLimitKind,
) -> pb::RunStopReason {
    match limit {
        agent_sdk_foundation::types::BudgetLimitKind::TotalTokens => {
            pb::RunStopReason::BudgetTotalTokens
        }
        agent_sdk_foundation::types::BudgetLimitKind::CostUsd => pb::RunStopReason::BudgetCostUsd,
    }
}

fn map_lifecycle_event_payload(event: &AgentEvent) -> RpcResult<pb::event_envelope::Event> {
    match event {
        AgentEvent::TurnComplete {
            turn,
            usage,
            emitter_task_id,
        } => Ok(pb::event_envelope::Event::TurnComplete(
            pb::TurnCompleteEvent {
                turn: map_u64(*turn, "turn")?,
                usage: Some(map_token_usage(usage)),
                emitter_task_id: emitter_task_id.clone(),
            },
        )),
        // Both terminal run-completion markers project to `DoneEvent`. See
        // `map_done_event` for why `BudgetExceeded` reuses the `done` frame.
        AgentEvent::Done {
            thread_id,
            total_turns,
            total_usage,
            duration,
            estimated_cost_usd,
            emitter_task_id,
        } => map_done_event(
            thread_id,
            *total_turns,
            total_usage,
            *duration,
            pb::RunStopReason::Completed,
            *estimated_cost_usd,
            emitter_task_id.clone(),
        ),
        AgentEvent::BudgetExceeded {
            thread_id,
            total_turns,
            total_usage,
            duration,
            estimated_cost_usd,
            limit,
            emitter_task_id,
        } => map_done_event(
            thread_id,
            *total_turns,
            total_usage,
            *duration,
            budget_stop_reason(*limit),
            *estimated_cost_usd,
            emitter_task_id.clone(),
        ),
        AgentEvent::Error {
            message,
            recoverable,
            emitter_task_id,
        } => Ok(pb::event_envelope::Event::Error(pb::ErrorEvent {
            message: message.clone(),
            recoverable: *recoverable,
            emitter_task_id: emitter_task_id.clone(),
        })),
        AgentEvent::Cancelled {
            turn,
            usage,
            emitter_task_id,
        } => Ok(pb::event_envelope::Event::Cancelled(pb::CancelledEvent {
            turn: map_u64(*turn, "turn")?,
            usage: Some(map_token_usage(usage)),
            emitter_task_id: emitter_task_id.clone(),
        })),
        AgentEvent::ContextCompacted {
            original_count,
            new_count,
            original_tokens,
            new_tokens,
        } => Ok(pb::event_envelope::Event::ContextCompacted(
            pb::ContextCompactedEvent {
                original_count: map_u64(*original_count, "original_count")?,
                new_count: map_u64(*new_count, "new_count")?,
                original_tokens: map_u64(*original_tokens, "original_tokens")?,
                new_tokens: map_u64(*new_tokens, "new_tokens")?,
            },
        )),
        AgentEvent::SubagentProgress { .. } => map_subagent_progress_payload(event),
        AgentEvent::AutoRetryStart { .. } | AgentEvent::AutoRetryEnd { .. } => {
            map_auto_retry_payload(event)
        }
        _ => Err(Status::internal("unsupported event variant").into()),
    }
}

/// Map the auto-retry signalling events onto their protobuf payloads.
/// Split from [`map_lifecycle_event_payload`] purely for length; any
/// other variant is an internal error.
fn map_auto_retry_payload(event: &AgentEvent) -> RpcResult<pb::event_envelope::Event> {
    match event {
        AgentEvent::AutoRetryStart {
            attempt,
            max_attempts,
            delay_ms,
            error_message,
        } => Ok(pb::event_envelope::Event::AutoRetryStart(
            pb::AutoRetryStartEvent {
                attempt: *attempt,
                max_attempts: *max_attempts,
                delay_ms: *delay_ms,
                error_message: error_message.clone(),
            },
        )),
        AgentEvent::AutoRetryEnd {
            attempt,
            success,
            final_error,
        } => Ok(pb::event_envelope::Event::AutoRetryEnd(
            pb::AutoRetryEndEvent {
                attempt: *attempt,
                success: *success,
                final_error: final_error.clone(),
            },
        )),
        _ => Err(Status::internal("unsupported event variant").into()),
    }
}

/// Map [`AgentEvent::SubagentProgress`] onto its protobuf payload.
/// Split from [`map_lifecycle_event_payload`] purely for length; any
/// other variant is an internal error.
fn map_subagent_progress_payload(event: &AgentEvent) -> RpcResult<pb::event_envelope::Event> {
    let AgentEvent::SubagentProgress {
        subagent_id,
        subagent_name,
        nickname,
        child_thread_id,
        child_root_task_id,
        subagent_task_id,
        max_turns,
        current_turn,
        model,
        tool_name,
        tool_context,
        completed,
        success,
        tool_count,
        total_tokens,
    } = event
    else {
        return Err(Status::internal("expected a subagent-progress event").into());
    };
    Ok(pb::event_envelope::Event::SubagentProgress(
        pb::SubagentProgressEvent {
            subagent_id: subagent_id.clone(),
            subagent_name: subagent_name.clone(),
            nickname: nickname.clone(),
            child_thread_id: child_thread_id.as_ref().map(ToString::to_string),
            child_root_task_id: child_root_task_id.clone(),
            subagent_task_id: subagent_task_id.clone(),
            max_turns: *max_turns,
            current_turn: *current_turn,
            model: model.clone(),
            tool_name: tool_name.clone(),
            tool_context: tool_context.clone(),
            completed: *completed,
            success: *success,
            tool_count: *tool_count,
            total_tokens: *total_tokens,
        },
    ))
}

fn map_u64<T>(value: T, field: &'static str) -> RpcResult<u64>
where
    T: TryInto<u64>,
{
    value
        .try_into()
        .map_err(|_| Status::internal(format!("{field} out of range for protobuf")).into())
}

/// Close reason implied by a terminal lifecycle event, or `None` for
/// every non-terminal event. `Done` and `BudgetExceeded` complete the
/// followed work (the latter is emitted in place of `Done` when a budget
/// trips, so a budget-terminated thread closes cleanly instead of hanging);
/// `Cancelled` is the durable marker committed by an effective cancel
/// (see `CancelledEvent` in `events.proto`) — all three close the stream.
const fn terminal_close_reason(
    event: &agent_server::CommittedEvent,
) -> Option<pb::StreamCloseReason> {
    match event.event {
        AgentEvent::Done { .. } | AgentEvent::BudgetExceeded { .. } => {
            Some(pb::StreamCloseReason::ThreadCompleted)
        }
        AgentEvent::Cancelled { .. } => Some(pb::StreamCloseReason::TurnCancelled),
        _ => None,
    }
}

/// [`terminal_close_reason`], confirmed against the task journal: a
/// `ThreadCompleted` frame only closes the stream when it is the
/// thread's own terminal frame, not a
/// superseded attempt's. A cancelled predecessor's late full-turn
/// commit sequences its `Done` after the promoted successor's `Start`
/// — closing on that foreign frame would cut the follower off before
/// the successor's retry boundary and answer, which the requeue path
/// explicitly promises to deliver.
///
/// Staleness is a property of the FRAME versus the thread's committed
/// turns — determined from monotonic durable state only, never from
/// task rows: a frame whose `total_turns` is BEHIND the thread's
/// latest checkpoint is stale outright — later
/// turns were already durably committed (the collision successor's
/// boundary/answer/`Done`, or simply newer turns during backfill),
/// and closing at the stale frame would truncate their delivery. A
/// frame AT the latest committed turn always closes.
///
/// Task-state inference beyond the emitter's own cancellation is
/// deliberately absent. Any discriminator built on mutable task state
/// (any-active-root, latest-committer, frame-turn + active-root) opens
/// a timing hole, because task rows mutate independently of event
/// delivery: the emitter mid-transition reads as a successor (stream
/// hangs), a queued root promoted before publish reads as a collision
/// successor (a VALID `Done` becomes timing-dependent), and a successor
/// leaving the slot un-checkpointed re-admits the stale close.
/// Checkpoint turns only advance, so the monotonic arm is
/// deterministic and can neither hang a stream nor suppress a genuine
/// tail frame.
///
/// # Emitter attribution
///
/// The monotonic rule alone cannot see a superseded frame delivered
/// BEFORE its successor commits anything durable: a cancelled root's
/// late full-turn salvage lands `Done` at the very turn the checkpoint
/// head holds, so it reads as the thread's own tail. The frame's
/// `emitter_task_id` closes that window by answering a question the
/// turn number cannot: *was this frame's emitter durably superseded?*
///
/// Only one status is acted on — `Cancelled` — and only ever to
/// SUPPRESS a close, never to force one. That is what makes the read
/// race-free where inference over live state is not:
///
/// - `Cancelled` is terminal/absorbing, so once it reads back it stays
///   that way; there is no transient window to sample wrong.
/// - Reading anything else means the close is correct as of the read; a
///   cancel landing immediately after commits its own marker, which the
///   follower still receives — the same benign race that already exists
///   between a `Done` and the next submit.
/// - The cancelled row must belong to the thread being followed. A fork
///   inherits the source's frames verbatim, emitter ids included, so a
///   cancel on the SOURCE thread must not suppress a close on the
///   independent destination (see [`emitter_was_cancelled`]).
///
/// A cancelled emitter's terminal frame is one of two things, and the
/// gate deliberately does not try to tell them apart:
///
/// - **Late salvage** of superseded work — the cancelled root's turn
///   commit landing after the cancel took its slot.
/// - **A genuine, fully-billed final answer** whose task was cancelled
///   at the finish line. Completed-turn commits are deliberately left
///   unfenced against a concurrent cancel precisely so such an answer is
///   never discarded, so this ordering is reachable by design.
///
/// Suppression is safe for both because the cancel transaction has
/// ALREADY journaled the marker that closes the stream, and neither
/// frame is dropped from the journal — the answer is still delivered,
/// only its close reason shifts from `ThreadCompleted` (on the frame) to
/// `TurnCancelled` (on the marker). Trying to distinguish them would
/// mean inferring over exactly the mutable state this gate refuses to
/// read.
///
/// Suppression keeps the follower attached, which is what the requeue
/// path promises: it then receives the successor's retry boundary, fresh
/// `Start`, and genuine `Done`, and closes on that.
///
/// # Cancels with no successor
///
/// When nothing was queued behind the cancelled root, no successor ever
/// commits, so a follower positioned PAST the marker (a reconnect with a
/// later cursor, or a replay whose tail is the suppressed frame rather
/// than the marker) receives no further lifecycle close and idles until
/// the client disconnects or a new turn is submitted. That is the
/// documented salvage contract, not a defect: `CancelledEvent` in
/// `events.proto` states that a cursor past the marker may observe
/// trailing salvage "with no further lifecycle close", and names the
/// marker the authoritative end of the cancelled turn. An idle follow is
/// a legal follower state; a `ThreadCompleted` on a cancelled root's
/// frame would be a lie about how the thread ended.
///
/// Every fallback (no emitter id, row missing, read error) reproduces the
/// historical behavior, so no id-less frame — an old journal row, an
/// embedded run — changes disposition in either direction.
///
/// `TurnCancelled` closes are NOT gated: `cancel_tree` promotes the
/// queued successor in the same transaction as its marker, so an
/// active root is the NORMAL state when the marker is delivered, and
/// the marker's whole purpose is to release followers of the
/// cancelled turn. A journal read failure keeps the historical close
/// (never strand a client on a dead thread).
async fn confirmed_close_reason(
    shared: &GrpcShared,
    thread_id: &ThreadId,
    event: &agent_server::CommittedEvent,
) -> Option<pb::StreamCloseReason> {
    let reason = terminal_close_reason(event)?;
    if !matches!(reason, pb::StreamCloseReason::ThreadCompleted) {
        return Some(reason);
    }
    if frame_is_behind_committed_head(shared, thread_id, event).await {
        return None;
    }
    if emitter_was_cancelled(shared, thread_id, event).await {
        return None;
    }
    Some(reason)
}

/// Is this terminal frame's turn behind the thread's latest committed
/// checkpoint? Later turns are already durable (a collision successor's
/// boundary/answer/`Done`, or simply newer turns during backfill), so
/// closing here would truncate their delivery.
///
/// A checkpoint read failure reports `false` — keep the historical
/// close rather than strand a client on a dead thread.
async fn frame_is_behind_committed_head(
    shared: &GrpcShared,
    thread_id: &ThreadId,
    event: &agent_server::CommittedEvent,
) -> bool {
    let frame_turn = match &event.event {
        AgentEvent::Done { total_turns, .. } | AgentEvent::BudgetExceeded { total_turns, .. } => {
            u32::try_from(*total_turns).ok()
        }
        _ => None,
    };
    let Some(frame_turn) = frame_turn else {
        return false;
    };
    let latest = match shared
        .stores
        .checkpoint_store
        .get_latest_by_thread(thread_id)
        .await
    {
        Ok(latest) => latest,
        Err(error) => {
            warn!(
                thread_id = %thread_id,
                error = %format!("{error:#}"),
                "terminal-close checkpoint read failed; keeping the close",
            );
            return false;
        }
    };
    latest.is_some_and(|latest| latest.turn_number > frame_turn)
}

/// Did the task that committed this frame get cancelled ON THE THREAD
/// BEING FOLLOWED? Only a row that reads back `Cancelled` and is bound
/// to `thread_id` suppresses the close; an unstamped frame, a missing
/// row, a foreign-thread emitter, and a read failure all report `false`
/// and leave the historical disposition intact.
///
/// The thread check is what keeps forks correct. `ForkThread` re-commits
/// the source's terminal frames verbatim onto the destination — their
/// provenance, emitter id included, travels with them by design — so a
/// forked `Done` names a task that never ran on the thread being
/// followed. Cancelling that source task supersedes work on the SOURCE
/// thread and says nothing about the fork, which has its own independent
/// history (no marker, no successor); suppressing there would idle a
/// follower that must close. A task's thread binding is immutable, so
/// this comparison reads no mutable state and leaves the absorbing-status
/// argument intact.
async fn emitter_was_cancelled(
    shared: &GrpcShared,
    thread_id: &ThreadId,
    event: &agent_server::CommittedEvent,
) -> bool {
    let Some(emitter) = event.event.emitter_task_id() else {
        return false;
    };
    let emitter = agent_server::journal::task::AgentTaskId::from_string(emitter);
    match shared.stores.task_store.get(&emitter).await {
        Ok(Some(task)) => {
            task.thread_id == *thread_id
                && task.status == agent_server::journal::task::TaskStatus::Cancelled
        }
        Ok(None) => false,
        Err(error) => {
            warn!(
                task_id = %emitter,
                error = %format!("{error:#}"),
                "terminal-close emitter read failed; keeping the close",
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::*;
    use crate::runtime::{
        AllowAllConfirmationPolicy, ExecutionRuntime, NoopToolExecutor, StaticProviderResolver,
        ToolCallExecutor,
    };
    #[cfg(feature = "postgres")]
    use agent_sdk_foundation::ThreadId;
    use agent_sdk_foundation::llm::{
        ChatOutcome, ChatRequest, ChatResponse, StopReason, Tool, Usage,
    };
    use agent_sdk_foundation::{
        AgentContinuation, AgentState, ContinuationEnvelope, ListenExecutionContext,
    };
    use agent_sdk_providers::LlmProvider;
    #[cfg(feature = "postgres")]
    use agent_server::AgentTaskId;
    use agent_server::journal::SubagentInvocationSpawn;
    use agent_server::journal::recovery::RecoveryRecord;
    use agent_server::journal::store::{AgentTaskStore, SubmitRootTurnOutcome};
    use agent_server::journal::task::{ChildSpawnSpec, SuspensionPayload};
    use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
    use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
    use anyhow::{Context, Result, anyhow, bail, ensure};
    use async_trait::async_trait;
    use serde_json::json;
    #[cfg(feature = "postgres")]
    use sqlx::{Connection, PgConnection};
    use tonic::transport::Channel;
    #[cfg(feature = "postgres")]
    use uuid::Uuid;

    type ControlClient = pb::agent_control_service_client::AgentControlServiceClient<Channel>;
    type EventClient = pb::agent_event_service_client::AgentEventServiceClient<Channel>;
    type StreamItem = pb::stream_thread_events_response::Item;
    type EventPayload = pb::event_envelope::Event;

    #[test]
    fn map_message_omits_opaque_reasoning_from_public_conversation() -> Result<()> {
        let message = llm::Message::assistant_with_content(vec![
            ContentBlock::OpaqueReasoning {
                provider: "test-provider".to_owned(),
                data: json!({"encrypted_content": "opaque-secret"}),
            },
            ContentBlock::Text {
                text: "visible".to_owned(),
            },
        ]);

        let mapped = map_message(&message).map_err(|error| anyhow!(error.to_string()))?;
        let Some(pb::conversation_message::Content::Blocks(blocks)) = mapped.content else {
            bail!("assistant block content should map to a block list");
        };
        assert_eq!(blocks.items.len(), 1);
        assert!(matches!(
            blocks.items.first().and_then(|item| item.block.as_ref()),
            Some(pb::conversation_content_block::Block::Text(text)) if text.text == "visible"
        ));
        Ok(())
    }

    struct ScriptedProvider {
        responses: Mutex<VecDeque<ChatResponse>>,
    }

    impl ScriptedProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            let response = {
                let mut responses = self
                    .responses
                    .lock()
                    .map_err(|_| anyhow!("lock poisoned"))?;
                responses
                    .pop_front()
                    .context("no scripted response remaining")?
            };
            Ok(ChatOutcome::Success(response))
        }

        fn model(&self) -> &'static str {
            "mock-model"
        }

        fn provider(&self) -> &'static str {
            "mock"
        }
    }

    #[derive(Clone)]
    struct ProgressToolExecutor {
        result: ToolResult,
        emit_progress: bool,
    }

    #[async_trait]
    impl ToolCallExecutor for ProgressToolExecutor {
        async fn execute_tool_call(
            &self,
            bootstrap: &agent_server::ToolTaskBootstrap,
            collector: agent_server::worker::ToolEventCollector,
            _cancel: CancellationToken,
        ) -> Result<ToolResult> {
            if self.emit_progress {
                collector.emit(AgentEvent::tool_progress(
                    &bootstrap.tool_call.id,
                    &bootstrap.tool_call.name,
                    &bootstrap.tool_call.display_name,
                    "running",
                    "in progress",
                    Some(json!({"step": 1})),
                ));
            }
            Ok(self.result.clone())
        }
    }

    fn mock_definition(tools: Vec<Tool>) -> AgentDefinition {
        AgentDefinition {
            provider: "mock".into(),
            model: "mock-model".into(),
            system_prompt: "test".into(),
            max_tokens: 512,
            tools,
            thinking: ThinkingPolicy::default(),
            tools_fn: None,
            policy: RuntimePolicy::server_default(),
        }
    }

    fn text_response(id: &str, text: &str) -> ChatResponse {
        ChatResponse {
            id: id.into(),
            content: vec![ContentBlock::Text { text: text.into() }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }
    }

    fn tool_use_response(
        id: &str,
        tool_call_id: &str,
        name: &str,
        input: serde_json::Value,
    ) -> ChatResponse {
        ChatResponse {
            id: id.into(),
            content: vec![ContentBlock::ToolUse {
                id: tool_call_id.into(),
                name: name.into(),
                input,
                thought_signature: None,
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 12,
                output_tokens: 6,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        }
    }

    fn runtime_with(
        provider: Arc<dyn LlmProvider>,
        tool_executor: Arc<dyn ToolCallExecutor>,
    ) -> Result<Arc<ExecutionRuntime>> {
        let resolver = Arc::new(StaticProviderResolver::new());
        resolver.set_fallback(provider)?;
        Ok(Arc::new(ExecutionRuntime::new(
            resolver,
            tool_executor,
            Arc::new(AllowAllConfirmationPolicy),
        )))
    }

    fn event_test_shared() -> Result<Arc<GrpcShared>> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let stores = StoreRegistry::in_memory(registry);
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(Vec::new())),
            Arc::new(NoopToolExecutor),
        )?;
        Ok(Arc::new(GrpcShared::new(
            stores,
            runtime,
            HealthSurface::shared(),
            CancellationToken::new(),
            time::Duration::seconds(30),
            // The worker default: 30s lease / 10s heartbeat (ADR-0003 I4).
            std::time::Duration::from_secs(10),
            AdmissionConfig::default(),
        )))
    }

    async fn seed_events(shared: &GrpcShared, thread_id: &ThreadId, count: u32) -> Result<()> {
        let now = OffsetDateTime::UNIX_EPOCH;
        for i in 0..count {
            shared
                .stores
                .event_repo
                .commit_event(
                    thread_id,
                    agent_sdk_foundation::events::AgentEvent::text(format!("m{i}"), "x"),
                    now,
                )
                .await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn thread_snapshot_reports_boundaries_without_full_scan() -> Result<()> {
        // Regression: thread_snapshot must read the two boundary
        // sequences from cheap watermark/min queries rather than loading
        // the whole journal.
        let shared = event_test_shared()?;
        let thread_id = ThreadId::from_string("t-snapshot-bounds");
        let now = OffsetDateTime::UNIX_EPOCH;
        let thread = shared
            .stores
            .thread_store
            .get_or_create(&thread_id, now)
            .await?;
        seed_events(&shared, &thread_id, 4).await?;

        let snapshot = shared.thread_snapshot(&thread).await?;
        assert_eq!(snapshot.latest_event_sequence, Some(3));
        assert_eq!(snapshot.earliest_available_event_sequence, Some(0));
        Ok(())
    }

    #[tokio::test]
    async fn fetch_replay_events_reads_only_the_suffix() -> Result<()> {
        let shared = event_test_shared()?;
        let thread_id = ThreadId::from_string("t-replay-suffix");
        seed_events(&shared, &thread_id, 6).await?;
        let latest_available = Some(5);

        // Reconnect after sequence 3 → only the missed suffix 4,5.
        let suffix =
            fetch_replay_events(shared.as_ref(), &thread_id, Some(3), latest_available).await?;
        assert_eq!(
            suffix
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![4, 5],
        );

        // Replay-from-start (None) is the only path that reads it all.
        let all = fetch_replay_events(shared.as_ref(), &thread_id, None, latest_available).await?;
        assert_eq!(all.len(), 6);
        Ok(())
    }

    #[tokio::test]
    async fn fetch_missed_range_recovers_skipped_events() -> Result<()> {
        let shared = event_test_shared()?;
        let thread_id = ThreadId::from_string("t-missed-range");
        seed_events(&shared, &thread_id, 6).await?;

        // Frontier at sequence 2, an out-of-order live event for sequence
        // 5 arrives → the dropped gap is [3, 4].
        let missed = fetch_missed_range(shared.as_ref(), &thread_id, Some(2), 5).await?;
        assert_eq!(
            missed
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![3, 4],
        );

        // Nothing delivered yet and event 3 arrives → prefix [0, 1, 2].
        let prefix = fetch_missed_range(shared.as_ref(), &thread_id, None, 3).await?;
        assert_eq!(
            prefix
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1, 2],
        );
        Ok(())
    }

    async fn connect_clients(endpoint: &str) -> Result<(ControlClient, EventClient)> {
        let channel = Channel::from_shared(endpoint.to_owned())
            .context("building channel endpoint")?
            .connect()
            .await
            .context("connecting channel")?;
        Ok((
            ControlClient::new(channel.clone()),
            EventClient::new(channel),
        ))
    }

    fn text_input(text: &str) -> pb::UserInputItem {
        pb::UserInputItem {
            item: Some(pb::user_input_item::Item::Text(text.into())),
        }
    }

    #[cfg(feature = "postgres")]
    fn message_text(message: &pb::ConversationMessage) -> Option<&str> {
        match message.content.as_ref() {
            Some(pb::conversation_message::Content::Text(text)) => Some(text.as_str()),
            Some(pb::conversation_message::Content::Blocks(blocks)) => blocks
                .items
                .iter()
                .find_map(|item| match item.block.as_ref() {
                    Some(pb::conversation_content_block::Block::Text(text)) => {
                        Some(text.text.as_str())
                    }
                    _ => None,
                }),
            _ => None,
        }
    }

    #[cfg(feature = "postgres")]
    fn drop_test_schema(database_url: String, schema: String) {
        let _ = std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            runtime.block_on(async move {
                let Ok(mut conn) = PgConnection::connect(&database_url).await else {
                    return;
                };
                let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                    "DROP SCHEMA IF EXISTS {schema} CASCADE"
                )))
                .execute(&mut conn)
                .await;
            });
        })
        .join();
    }

    // ── Postgres-specific test infrastructure and integration tests ───
    #[cfg(feature = "postgres")]
    struct PostgresTestSchema {
        schema: String,
        database_url: String,
    }

    #[cfg(feature = "postgres")]
    impl Drop for PostgresTestSchema {
        fn drop(&mut self) {
            drop_test_schema(self.database_url.clone(), self.schema.clone());
        }
    }

    #[cfg(feature = "postgres")]
    async fn postgres_test_config() -> Result<Option<(ServiceConfig, PostgresTestSchema)>> {
        let Ok(database_url) =
            std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        else {
            return Ok(None);
        };

        let schema = format!("eng_7999_{}", Uuid::new_v4().simple());
        let mut admin = PgConnection::connect(&database_url)
            .await
            .context("connect postgres admin for grpc tests")?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut admin)
            .await
            .with_context(|| format!("create grpc test schema {schema}"))?;

        Ok(Some((
            ServiceConfig {
                storage: crate::config::StorageConfig {
                    backend: crate::config::StorageBackend::Postgres,
                    postgres: crate::config::PostgresStorageConfig {
                        database_url: Some(database_url.clone()),
                        schema: Some(schema.clone()),
                        max_connections: 8,
                    },
                },
                ..ServiceConfig::default()
            },
            PostgresTestSchema {
                schema,
                database_url,
            },
        )))
    }

    #[cfg(feature = "postgres")]
    async fn inspection_stores(config: &ServiceConfig) -> Result<StoreRegistry> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let stores = StoreRegistry::from_config(&config.storage, registry)
            .context("build inspection stores")?;
        stores
            .initialize()
            .await
            .context("initialize inspection stores")?;
        Ok(stores)
    }

    #[cfg(feature = "postgres")]
    fn empty_definition_registry() -> Arc<dyn AgentDefinitionRegistry> {
        Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )))
    }

    #[cfg(feature = "postgres")]
    fn postgres_text_runtime(response_id: &str, text: &str) -> Result<Arc<ExecutionRuntime>> {
        runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response(
                response_id,
                text,
            )])),
            Arc::new(NoopToolExecutor),
        )
    }

    #[cfg(feature = "postgres")]
    async fn start_postgres_daemon_with_text(
        config: ServiceConfig,
        response_id: &str,
        text: &str,
    ) -> Result<LocalDaemon> {
        LocalDaemon::start(
            config,
            empty_definition_registry(),
            postgres_text_runtime(response_id, text)?,
        )
        .await
    }

    async fn create_thread(control: &mut ControlClient, request_id: &str) -> Result<String> {
        let response = control
            .create_thread(pb::CreateThreadRequest {
                request_id: request_id.into(),
                thread_id: String::new(),
            })
            .await
            .context("create_thread rpc")?
            .into_inner();
        let thread = response
            .thread
            .context("create_thread response missing thread")?;
        let snapshot = thread
            .thread
            .context("create_thread response missing snapshot")?;
        Ok(snapshot.thread_id)
    }

    async fn submit_text_work(
        control: &mut ControlClient,
        request_id: &str,
        thread_id: &str,
        text: &str,
    ) -> Result<pb::TaskSnapshot> {
        let response = control
            .submit_thread_work(pb::SubmitThreadWorkRequest {
                request_id: request_id.into(),
                thread_id: thread_id.into(),
                input: vec![text_input(text)],
                caller_metadata: String::new(),
            })
            .await
            .context("submit_thread_work rpc")?
            .into_inner();
        response.task.context("submit_thread_work missing task")
    }

    async fn open_stream(
        events: &mut EventClient,
        thread_id: &str,
        after_sequence: Option<u64>,
        follow_mode: pb::FollowMode,
    ) -> Result<tonic::Streaming<pb::StreamThreadEventsResponse>> {
        let response = events
            .stream_thread_events(pb::StreamThreadEventsRequest {
                thread_id: thread_id.into(),
                after_sequence,
                follow_mode: follow_mode as i32,
            })
            .await
            .context("stream_thread_events rpc")?;
        Ok(response.into_inner())
    }

    async fn next_stream_item(
        stream: &mut tonic::Streaming<pb::StreamThreadEventsResponse>,
    ) -> Result<pb::StreamThreadEventsResponse> {
        tokio::time::timeout(Duration::from_secs(10), stream.message())
            .await
            .context("timed out waiting for event stream item")?
            .context("stream message error")?
            .context("event stream closed unexpectedly")
    }

    async fn collect_until_closed(
        stream: &mut tonic::Streaming<pb::StreamThreadEventsResponse>,
    ) -> Result<Vec<pb::StreamThreadEventsResponse>> {
        let mut items = Vec::new();
        loop {
            let item = next_stream_item(stream).await?;
            let terminal = matches!(
                item.item.as_ref(),
                Some(
                    StreamItem::Closed(_)
                        | StreamItem::RetentionGap(_)
                        | StreamItem::ReplayRequired(_)
                )
            );
            items.push(item);
            if terminal {
                return Ok(items);
            }
        }
    }

    fn event_sequences(items: &[pb::StreamThreadEventsResponse]) -> Vec<u64> {
        items
            .iter()
            .filter_map(|item| match item.item.as_ref() {
                Some(StreamItem::Event(event)) => Some(event.sequence),
                _ => None,
            })
            .collect()
    }

    fn assert_contiguous_sequences(sequences: &[u64]) {
        for (index, sequence) in sequences.iter().enumerate() {
            assert_eq!(*sequence, index as u64, "sequence gap at index {index}");
        }
    }

    async fn wait_for<F, Fut, T>(mut check: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<Option<T>>>,
    {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(value) = check().await? {
                return Ok(value);
            }
            if tokio::time::Instant::now() >= deadline {
                bail!("timed out waiting for condition");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn list_thread_tasks(
        control: &mut ControlClient,
        thread_id: &str,
        context_label: &'static str,
    ) -> Result<Vec<pb::TaskSnapshot>> {
        Ok(control
            .list_thread_tasks(pb::ListThreadTasksRequest {
                thread_id: thread_id.into(),
            })
            .await
            .with_context(|| format!("{context_label} list_thread_tasks"))?
            .into_inner()
            .tasks)
    }

    #[cfg(feature = "postgres")]
    async fn wait_for_committed_turns(
        control: &ControlClient,
        thread_id: &str,
        expected_turns: u32,
        context_label: &'static str,
    ) -> Result<()> {
        let thread_id = thread_id.to_owned();
        wait_for(|| {
            let mut control = control.clone();
            let thread_id = thread_id.clone();
            async move {
                let response = control
                    .get_thread(pb::GetThreadRequest { thread_id })
                    .await
                    .with_context(|| context_label.to_owned())?
                    .into_inner();
                let committed_turns = response
                    .thread
                    .and_then(|view| view.thread)
                    .map(|thread| thread.committed_turns)
                    .unwrap_or_default();
                Ok((committed_turns == expected_turns).then_some(()))
            }
        })
        .await
    }

    #[cfg(feature = "postgres")]
    struct PersistedPostgresState {
        inspection: StoreRegistry,
        thread_key: ThreadId,
        persisted_event_sequences: Vec<u64>,
    }

    #[cfg(feature = "postgres")]
    async fn run_first_postgres_daemon_pass(daemon: &LocalDaemon) -> Result<(String, String)> {
        let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;
        let thread_id = create_thread(&mut control, "create-postgres-thread").await?;
        let _task = submit_text_work(
            &mut control,
            "submit-postgres-turn-1",
            &thread_id,
            "turn one",
        )
        .await?;

        let mut live_stream = open_stream(
            &mut events,
            &thread_id,
            None,
            pb::FollowMode::ReplayAndFollow,
        )
        .await?;
        let last_seen = loop {
            let item = next_stream_item(&mut live_stream).await?;
            if let Some(StreamItem::Event(event)) = item.item {
                break event.sequence;
            }
        };
        drop(live_stream);

        wait_for_committed_turns(
            &control,
            &thread_id,
            1,
            "poll get_thread during postgres replay",
        )
        .await?;

        let mut replay_stream = open_stream(
            &mut events,
            &thread_id,
            Some(last_seen),
            pb::FollowMode::ReplayOnly,
        )
        .await?;
        let replay_items = collect_until_closed(&mut replay_stream).await?;
        let replayed = event_sequences(&replay_items);
        assert!(
            replayed.iter().all(|sequence| *sequence > last_seen),
            "postgres replay returned an event at or before the last seen sequence",
        );

        let messages = control
            .get_thread_messages(pb::GetThreadMessagesRequest {
                thread_id: thread_id.clone(),
            })
            .await
            .context("get_thread_messages after first postgres turn")?
            .into_inner()
            .projection
            .context("missing postgres message projection after first turn")?;
        assert_eq!(messages.messages.len(), 2);

        let tasks = list_thread_tasks(&mut control, &thread_id, "postgres first run").await?;
        assert_eq!(tasks.len(), 1);
        Ok((thread_id, tasks[0].task_id.clone()))
    }

    #[cfg(feature = "postgres")]
    async fn inspect_first_postgres_restart_state(
        config: &ServiceConfig,
        thread_id: &str,
        first_task_id: &str,
    ) -> Result<PersistedPostgresState> {
        let inspection = inspection_stores(config).await?;
        let thread_key = ThreadId::from_string(thread_id.to_owned());
        let first_task_key = AgentTaskId::from_string(first_task_id.to_owned());

        let persisted_thread = inspection
            .thread_store
            .get(&thread_key)
            .await?
            .context("persisted postgres thread missing after restart")?;
        assert_eq!(persisted_thread.committed_turns, 1);

        let persisted_messages = inspection.message_store.get_history(&thread_key).await?;
        assert_eq!(persisted_messages.len(), 2);

        let checkpoints = inspection
            .checkpoint_store
            .list_by_thread(&thread_key)
            .await?;
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].turn_number, 1);

        let persisted_task = inspection
            .task_store
            .get(&first_task_key)
            .await?
            .context("persisted postgres task missing after restart")?;
        assert_eq!(
            persisted_task.status,
            agent_server::journal::task::TaskStatus::Completed,
        );

        let attempts = inspection
            .attempt_store
            .list_by_task(&first_task_key)
            .await?;
        assert_eq!(attempts.len(), 1);

        let fresh_process_events = inspection.event_repo.get_events(&thread_key).await?;
        let persisted_event_sequences = fresh_process_events
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>();
        assert!(
            !persisted_event_sequences.is_empty(),
            "fresh inspection stores should expose durable committed events across restart",
        );

        Ok(PersistedPostgresState {
            inspection,
            thread_key,
            persisted_event_sequences,
        })
    }

    #[cfg(feature = "postgres")]
    async fn assert_postgres_state_after_restart(
        control: &mut ControlClient,
        events: &mut EventClient,
        persisted: &PersistedPostgresState,
        thread_id: &str,
        first_task_id: &str,
    ) -> Result<()> {
        let thread_snapshot = control
            .get_thread(pb::GetThreadRequest {
                thread_id: thread_id.into(),
            })
            .await
            .context("get_thread after postgres restart")?
            .into_inner()
            .thread
            .context("missing thread view after postgres restart")?
            .thread
            .context("missing thread snapshot after postgres restart")?;
        assert_eq!(thread_snapshot.committed_turns, 1);

        let messages_before = control
            .get_thread_messages(pb::GetThreadMessagesRequest {
                thread_id: thread_id.into(),
            })
            .await
            .context("get_thread_messages after postgres restart")?
            .into_inner()
            .projection
            .context("missing message projection after postgres restart")?;
        assert_eq!(messages_before.messages.len(), 2);
        assert_eq!(
            message_text(&messages_before.messages[1]),
            Some("hello from postgres turn 1"),
        );

        let tasks_before = list_thread_tasks(control, thread_id, "postgres restart").await?;
        assert_eq!(tasks_before.len(), 1);
        assert_eq!(tasks_before[0].task_id, first_task_id);

        let mut replay_stream =
            open_stream(events, thread_id, None, pb::FollowMode::ReplayOnly).await?;
        let replay_after_restart = collect_until_closed(&mut replay_stream).await?;
        let replay_sequences = event_sequences(&replay_after_restart);
        assert_eq!(replay_sequences, persisted.persisted_event_sequences);
        assert!(matches!(
            replay_after_restart.last().and_then(|item| item.item.as_ref()),
            Some(StreamItem::Closed(closed))
                if closed.reason == pb::StreamCloseReason::ReplayExhausted as i32
        ));
        Ok(())
    }

    #[cfg(feature = "postgres")]
    async fn submit_second_postgres_turn(
        control: &mut ControlClient,
        events: &mut EventClient,
        thread_id: &str,
    ) -> Result<String> {
        let mut live_stream =
            open_stream(events, thread_id, None, pb::FollowMode::ReplayAndFollow).await?;
        let second_task =
            submit_text_work(control, "submit-postgres-turn-2", thread_id, "turn two").await?;
        let second_items = collect_until_closed(&mut live_stream).await?;
        let second_sequences = event_sequences(&second_items);
        assert_contiguous_sequences(&second_sequences);

        let thread_id_owned = thread_id.to_owned();
        let messages_after = wait_for(|| {
            let mut control = control.clone();
            let thread_id = thread_id_owned.clone();
            async move {
                let projection = control
                    .get_thread_messages(pb::GetThreadMessagesRequest { thread_id })
                    .await
                    .context("get_thread_messages after second postgres turn")?
                    .into_inner()
                    .projection;
                Ok(projection.filter(|projection| projection.messages.len() == 4))
            }
        })
        .await?;
        assert_eq!(messages_after.messages.len(), 4);
        assert_eq!(
            message_text(&messages_after.messages[1]),
            Some("hello from postgres turn 1"),
        );
        assert_eq!(
            message_text(&messages_after.messages[3]),
            Some("hello from postgres turn 2"),
        );

        let thread_after = control
            .get_thread(pb::GetThreadRequest {
                thread_id: thread_id.into(),
            })
            .await
            .context("get_thread after second postgres turn")?
            .into_inner()
            .thread
            .context("missing thread view after second postgres turn")?
            .thread
            .context("missing thread snapshot after second postgres turn")?;
        assert_eq!(thread_after.committed_turns, 2);

        let tasks_after = list_thread_tasks(control, thread_id, "postgres second run").await?;
        assert_eq!(tasks_after.len(), 2);
        Ok(second_task.task_id)
    }

    #[cfg(feature = "postgres")]
    async fn run_second_postgres_daemon_pass(
        daemon: &LocalDaemon,
        persisted: &PersistedPostgresState,
        thread_id: &str,
        first_task_id: &str,
    ) -> Result<String> {
        let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;
        assert_postgres_state_after_restart(
            &mut control,
            &mut events,
            persisted,
            thread_id,
            first_task_id,
        )
        .await?;
        submit_second_postgres_turn(&mut control, &mut events, thread_id).await
    }

    #[cfg(feature = "postgres")]
    async fn assert_second_postgres_persistence(
        persisted: &PersistedPostgresState,
        second_task_id: &str,
    ) -> Result<()> {
        let checkpoints_after = persisted
            .inspection
            .checkpoint_store
            .list_by_thread(&persisted.thread_key)
            .await?;
        assert_eq!(checkpoints_after.len(), 2);
        assert_eq!(
            checkpoints_after
                .last()
                .context("missing latest checkpoint after second turn")?
                .turn_number,
            2,
        );

        let second_task_key = AgentTaskId::from_string(second_task_id.to_owned());
        let second_task = persisted
            .inspection
            .task_store
            .get(&second_task_key)
            .await?
            .context("second postgres task missing from durable store")?;
        assert_eq!(
            second_task.status,
            agent_server::journal::task::TaskStatus::Completed,
        );

        let second_attempts = persisted
            .inspection
            .attempt_store
            .list_by_task(&second_task_key)
            .await?;
        assert_eq!(second_attempts.len(), 1);
        Ok(())
    }

    async fn wait_for_awaiting_confirmation(
        control: &ControlClient,
        thread_id: &str,
    ) -> Result<pb::TaskSnapshot> {
        let thread_id = thread_id.to_owned();
        wait_for(|| {
            let mut control = control.clone();
            let thread_id = thread_id.clone();
            async move {
                let tasks = list_thread_tasks(&mut control, &thread_id, "poll").await?;
                Ok(tasks
                    .into_iter()
                    .find(|task| task.status == pb::TaskStatus::AwaitingConfirmation as i32))
            }
        })
        .await
    }

    async fn wait_for_completed_tasks(control: &ControlClient, thread_id: &str) -> Result<()> {
        let thread_id = thread_id.to_owned();
        wait_for(|| {
            let mut control = control.clone();
            let thread_id = thread_id.clone();
            async move {
                let tasks = list_thread_tasks(&mut control, &thread_id, "poll final").await?;
                let all_completed = tasks.len() == 2
                    && tasks
                        .iter()
                        .all(|task| task.status == pb::TaskStatus::Completed as i32);
                Ok(all_completed.then_some(()))
            }
        })
        .await
    }

    async fn collect_until_confirmation(
        stream: &mut tonic::Streaming<pb::StreamThreadEventsResponse>,
    ) -> Result<(Vec<pb::StreamThreadEventsResponse>, u64)> {
        let mut items = Vec::new();
        let mut last_seen_sequence = None;
        loop {
            let item = next_stream_item(stream).await?;
            let saw_confirmation = matches!(
                item.item.as_ref(),
                Some(StreamItem::Event(event))
                    if matches!(
                        event.event.as_ref(),
                        Some(EventPayload::ToolRequiresConfirmation(_))
                    )
            );
            if let Some(StreamItem::Event(event)) = item.item.as_ref() {
                last_seen_sequence = Some(event.sequence);
            }
            items.push(item);
            if saw_confirmation {
                return Ok((
                    items,
                    last_seen_sequence.context("missing last seen sequence")?,
                ));
            }
        }
    }

    struct ConfirmationEventOrder {
        tool_start: usize,
        requires_confirmation: usize,
        requires_confirmation_display_name: String,
        tool_progress: usize,
        tool_end: usize,
        text: usize,
        done: usize,
    }

    fn confirmation_event_order(
        items: &[pb::StreamThreadEventsResponse],
    ) -> Result<ConfirmationEventOrder> {
        let mut saw_tool_start = None;
        let mut saw_tool_requires_confirmation = None;
        let mut saw_tool_requires_confirmation_display_name = None;
        let mut saw_tool_progress = None;
        let mut saw_tool_end = None;
        let mut saw_text = None;
        let mut saw_done = None;

        for (index, item) in items.iter().enumerate() {
            if let Some(StreamItem::Event(event)) = item.item.as_ref() {
                match event.event.as_ref().context("event payload missing")? {
                    EventPayload::ToolCallStart(_) => saw_tool_start = Some(index),
                    EventPayload::ToolRequiresConfirmation(payload) => {
                        saw_tool_requires_confirmation = Some(index);
                        saw_tool_requires_confirmation_display_name =
                            Some(payload.display_name.clone());
                    }
                    EventPayload::ToolProgress(_) => saw_tool_progress = Some(index),
                    EventPayload::ToolCallEnd(_) => saw_tool_end = Some(index),
                    EventPayload::Text(text) if text.text == "transfer complete" => {
                        saw_text = Some(index);
                    }
                    EventPayload::Done(_) => saw_done = Some(index),
                    _ => {}
                }
            }
        }

        Ok(ConfirmationEventOrder {
            tool_start: saw_tool_start.context("tool_call_start missing")?,
            requires_confirmation: saw_tool_requires_confirmation
                .context("tool_requires_confirmation missing")?,
            requires_confirmation_display_name: saw_tool_requires_confirmation_display_name
                .context("tool_requires_confirmation display name missing")?,
            tool_progress: saw_tool_progress.context("tool_progress missing")?,
            tool_end: saw_tool_end.context("tool_call_end missing")?,
            text: saw_text.context("final assistant text missing")?,
            done: saw_done.context("done event missing")?,
        })
    }

    fn assert_confirmation_event_order(items: &[pb::StreamThreadEventsResponse]) -> Result<()> {
        let sequences = event_sequences(items);
        assert_contiguous_sequences(&sequences);

        let order = confirmation_event_order(items)?;
        assert!(order.tool_start < order.requires_confirmation);
        assert!(order.requires_confirmation < order.tool_progress);
        assert!(order.tool_progress < order.tool_end);
        assert!(order.tool_end < order.text);
        assert!(order.text < order.done);
        assert_eq!(order.requires_confirmation_display_name, "Transfer");
        Ok(())
    }

    #[tokio::test]
    async fn local_daemon_streams_committed_text_turn_end_to_end() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response(
                "resp_text_1",
                "hello from daemon",
            )])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;
            let thread_id = create_thread(&mut control, "create-text-thread").await?;
            let _submitted =
                submit_text_work(&mut control, "submit-text-turn", &thread_id, "hi").await?;
            let mut stream = open_stream(
                &mut events,
                &thread_id,
                None,
                pb::FollowMode::ReplayAndFollow,
            )
            .await?;
            let items = collect_until_closed(&mut stream).await?;

            assert!(matches!(
                items.first().and_then(|item| item.item.as_ref()),
                Some(StreamItem::ReplayOpened(_))
            ));
            let sequences = event_sequences(&items);
            assert_contiguous_sequences(&sequences);

            let mut saw_start = false;
            let mut saw_text = false;
            let mut saw_done = false;
            for item in &items {
                if let Some(StreamItem::Event(event)) = item.item.as_ref() {
                    match event.event.as_ref().context("event payload missing")? {
                        EventPayload::Start(_) => saw_start = true,
                        EventPayload::Text(text) => {
                            saw_text = text.text == "hello from daemon";
                        }
                        EventPayload::Done(_) => saw_done = true,
                        _ => {}
                    }
                }
            }
            assert!(saw_start, "start event missing");
            assert!(saw_text, "text event missing");
            assert!(saw_done, "done event missing");

            let thread = control
                .get_thread(pb::GetThreadRequest {
                    thread_id: thread_id.clone(),
                })
                .await
                .context("get_thread rpc")?
                .into_inner()
                .thread
                .context("get_thread missing thread view")?;
            let snapshot = thread.thread.context("get_thread missing snapshot")?;
            assert_eq!(snapshot.committed_turns, 1);
            assert_eq!(snapshot.latest_event_sequence, sequences.last().copied());

            let messages = control
                .get_thread_messages(pb::GetThreadMessagesRequest {
                    thread_id: thread_id.clone(),
                })
                .await
                .context("get_thread_messages rpc")?
                .into_inner()
                .projection
                .context("missing message projection")?;
            assert_eq!(messages.messages.len(), 2);
            assert_eq!(messages.messages[0].role, pb::MessageRole::User as i32);
            assert_eq!(messages.messages[1].role, pb::MessageRole::Assistant as i32);

            let tasks = control
                .list_thread_tasks(pb::ListThreadTasksRequest { thread_id })
                .await
                .context("list_thread_tasks rpc")?
                .into_inner()
                .tasks;
            assert_eq!(tasks.len(), 1);
            assert_eq!(tasks[0].status, pb::TaskStatus::Completed as i32);
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    #[tokio::test]
    async fn local_daemon_replays_from_last_sequence_after_disconnect() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response(
                "resp_replay_1",
                "replay reply",
            )])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;
            let thread_id = create_thread(&mut control, "create-replay-thread").await?;
            let _task =
                submit_text_work(&mut control, "submit-replay-turn", &thread_id, "replay").await?;

            let mut live_stream = open_stream(
                &mut events,
                &thread_id,
                None,
                pb::FollowMode::ReplayAndFollow,
            )
            .await?;

            let last_seen = loop {
                let item = next_stream_item(&mut live_stream).await?;
                if let Some(StreamItem::Event(event)) = item.item {
                    break event.sequence;
                }
            };
            drop(live_stream);

            wait_for(|| {
                let mut control = control.clone();
                let thread_id = thread_id.clone();
                async move {
                    let response = control
                        .get_thread(pb::GetThreadRequest { thread_id })
                        .await
                        .context("poll get_thread")?
                        .into_inner();
                    let committed_turns = response
                        .thread
                        .and_then(|view| view.thread)
                        .map(|thread| thread.committed_turns)
                        .unwrap_or_default();
                    Ok((committed_turns == 1).then_some(()))
                }
            })
            .await?;

            let mut replay_stream = open_stream(
                &mut events,
                &thread_id,
                Some(last_seen),
                pb::FollowMode::ReplayOnly,
            )
            .await?;
            let items = collect_until_closed(&mut replay_stream).await?;
            let replayed = event_sequences(&items);
            assert!(
                replayed.iter().all(|sequence| *sequence > last_seen),
                "replay returned an event at or before the last seen sequence",
            );
            assert!(matches!(
                items.last().and_then(|item| item.item.as_ref()),
                Some(StreamItem::Closed(closed))
                    if closed.reason == pb::StreamCloseReason::ReplayExhausted as i32
            ));
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    #[tokio::test]
    #[cfg(feature = "postgres")]
    async fn local_daemon_postgres_restart_preserves_durable_core_and_exposes_event_gap()
    -> Result<()> {
        let Some((config, _schema_guard)) = postgres_test_config().await? else {
            return Ok(());
        };
        let daemon1 = start_postgres_daemon_with_text(
            config.clone(),
            "resp_pg_restart_1",
            "hello from postgres turn 1",
        )
        .await?;
        let first_run = run_first_postgres_daemon_pass(&daemon1).await;
        daemon1.stop().await?;
        let (thread_id, first_task_id) = first_run?;

        let persisted =
            inspect_first_postgres_restart_state(&config, &thread_id, &first_task_id).await?;

        let daemon2 = start_postgres_daemon_with_text(
            config.clone(),
            "resp_pg_restart_2",
            "hello from postgres turn 2",
        )
        .await?;
        let second_run =
            run_second_postgres_daemon_pass(&daemon2, &persisted, &thread_id, &first_task_id).await;
        daemon2.stop().await?;
        let second_task_id = second_run?;

        assert_second_postgres_persistence(&persisted, &second_task_id).await?;
        Ok(())
    }

    #[tokio::test]
    async fn local_daemon_confirms_tool_and_resumes_root_turn() -> Result<()> {
        let transfer_tool = Tool {
            name: "transfer".into(),
            description: "Transfer funds".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "amount": { "type": "number" } },
                "required": ["amount"]
            }),
            display_name: "Transfer".into(),
            tier: ToolTier::Confirm,
        };
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(vec![
            transfer_tool,
        ])));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![
                tool_use_response(
                    "resp_confirm_1",
                    "tool_call_1",
                    "transfer",
                    json!({"amount": 42}),
                ),
                text_response("resp_confirm_2", "transfer complete"),
            ])),
            Arc::new(ProgressToolExecutor {
                result: ToolResult::success("transfer ok"),
                emit_progress: true,
            }),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;
            let thread_id = create_thread(&mut control, "create-confirm-thread").await?;
            let _task = submit_text_work(
                &mut control,
                "submit-confirm-turn",
                &thread_id,
                "transfer 42",
            )
            .await?;
            let mut stream = open_stream(
                &mut events,
                &thread_id,
                None,
                pb::FollowMode::ReplayAndFollow,
            )
            .await?;
            let (mut items, replay_after) = collect_until_confirmation(&mut stream).await?;

            let awaiting_confirmation =
                wait_for_awaiting_confirmation(&control, &thread_id).await?;

            let decision = control
                .decide_confirmation(pb::DecideConfirmationRequest {
                    request_id: "approve-confirmation".into(),
                    thread_id: thread_id.clone(),
                    task_id: awaiting_confirmation.task_id.clone(),
                    decision: Some(pb::ConfirmationDecision {
                        decision: Some(pb::confirmation_decision::Decision::Approved(
                            pb::ApprovedConfirmation {},
                        )),
                    }),
                })
                .await
                .context("decide_confirmation rpc")?
                .into_inner();
            assert!(decision.task.is_some(), "decision response missing task");
            drop(stream);

            if let Err(error) = wait_for_completed_tasks(&control, &thread_id).await {
                let tasks =
                    list_thread_tasks(&mut control, &thread_id, "after completion timeout").await?;
                bail!("confirmation flow did not complete: {error:#}; tasks: {tasks:?}");
            }

            let mut replay_stream = open_stream(
                &mut events,
                &thread_id,
                Some(replay_after),
                pb::FollowMode::ReplayOnly,
            )
            .await?;
            items.extend(collect_until_closed(&mut replay_stream).await?);
            assert_confirmation_event_order(&items)?;

            let tasks = list_thread_tasks(&mut control, &thread_id, "rpc").await?;
            assert_eq!(tasks.len(), 2);
            assert!(
                tasks
                    .iter()
                    .all(|task| task.status == pb::TaskStatus::Completed as i32),
                "not every task completed after confirmation flow",
            );
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    // ── Store-decorator hook (issue #301) ──────────────────────────

    /// [`AgentTaskStore`] decorator that counts completion and
    /// admission calls and delegates everything to the wrapped store —
    /// the shape an embedding host uses with
    /// [`LocalDaemon::start_with_store_decorator`] to intercept
    /// subagent results.
    ///
    /// `fail_get` additionally turns every row read into an error, which
    /// is how the close gate's read-failure arm is exercised (an
    /// in-memory store never fails on its own).
    struct CountingTaskStore {
        inner: Arc<dyn AgentTaskStore>,
        complete_calls: Arc<AtomicUsize>,
        complete_with_result_calls: Arc<AtomicUsize>,
        submit_root_turn_idempotent_calls: Arc<AtomicUsize>,
        fail_get: bool,
    }

    #[async_trait]
    impl AgentTaskStore for CountingTaskStore {
        async fn insert(&self, task: AgentTask) -> Result<()> {
            self.inner.insert(task).await
        }
        async fn submit_root_turn(&self, task: AgentTask) -> Result<AgentTask> {
            self.inner.submit_root_turn(task).await
        }
        async fn submit_root_turn_idempotent(
            &self,
            params: SubmitRootTurnParams,
        ) -> std::result::Result<SubmitRootTurnOutcome, SubmitRootTurnError> {
            self.submit_root_turn_idempotent_calls
                .fetch_add(1, Ordering::SeqCst);
            self.inner.submit_root_turn_idempotent(params).await
        }
        async fn claim_idempotency(
            &self,
            request_id: &str,
            kind: IdempotencyKind,
            fingerprint: &[u8],
        ) -> Result<IdempotencyClaim> {
            self.inner
                .claim_idempotency(request_id, kind, fingerprint)
                .await
        }
        async fn record_idempotency(&self, record: IdempotencyRecord) -> Result<()> {
            self.inner.record_idempotency(record).await
        }
        async fn get(&self, id: &AgentTaskId) -> Result<Option<AgentTask>> {
            if self.fail_get {
                bail!("injected task-store read failure for {id}");
            }
            self.inner.get(id).await
        }
        async fn update(&self, task: AgentTask) -> Result<()> {
            self.inner.update(task).await
        }
        async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
            self.inner.list_by_thread(thread_id).await
        }
        async fn list_children(&self, parent_id: &AgentTaskId) -> Result<Vec<AgentTask>> {
            self.inner.list_children(parent_id).await
        }
        async fn list_by_status(&self, status: JournalTaskStatus) -> Result<Vec<AgentTask>> {
            self.inner.list_by_status(status).await
        }
        async fn active_root_for_thread(&self, thread_id: &ThreadId) -> Result<Option<AgentTask>> {
            self.inner.active_root_for_thread(thread_id).await
        }
        async fn list_queued_roots(&self, thread_id: &ThreadId) -> Result<Vec<AgentTask>> {
            self.inner.list_queued_roots(thread_id).await
        }
        async fn requeue_owned_task(
            &self,
            id: &agent_server::AgentTaskId,
            worker: &agent_server::WorkerId,
            lease: &agent_server::LeaseId,
            boundary: Option<agent_sdk_foundation::events::AgentEvent>,
            now: time::OffsetDateTime,
        ) -> Result<agent_server::journal::store::RequeueOutcome> {
            self.inner
                .requeue_owned_task(id, worker, lease, boundary, now)
                .await
        }
        async fn promote_next_queued_root(
            &self,
            thread_id: &ThreadId,
            now: OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.inner.promote_next_queued_root(thread_id, now).await
        }
        async fn try_acquire_task(
            &self,
            id: &AgentTaskId,
            worker: WorkerId,
            lease: LeaseId,
            expires_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.inner
                .try_acquire_task(id, worker, lease, expires_at, now)
                .await
        }
        async fn acquire_next_runnable(
            &self,
            worker: WorkerId,
            lease: LeaseId,
            expires_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.inner
                .acquire_next_runnable(worker, lease, expires_at, now)
                .await
        }
        async fn heartbeat_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            expires_at: OffsetDateTime,
            activity: Option<OffsetDateTime>,
            now: OffsetDateTime,
        ) -> Result<AgentTask> {
            self.inner
                .heartbeat_task(task_id, worker, lease, expires_at, activity, now)
                .await
        }
        async fn release_expired_leases(&self, now: OffsetDateTime) -> Result<Vec<RecoveryRecord>> {
            self.inner.release_expired_leases(now).await
        }
        async fn pause_on_children(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            child_count: u32,
            payload: SuspensionPayload,
            now: OffsetDateTime,
        ) -> Result<AgentTask> {
            self.inner
                .pause_on_children(task_id, worker, lease, child_count, payload, now)
                .await
        }
        async fn enqueue_steering_resume(
            &self,
            parent_id: &AgentTaskId,
            steering: Vec<ContentBlock>,
            now: OffsetDateTime,
        ) -> Result<Option<AgentTask>> {
            self.inner
                .enqueue_steering_resume(parent_id, steering, now)
                .await
        }
        async fn repark_after_steering(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            payload: SuspensionPayload,
            reattach: Vec<AgentTaskId>,
            now: OffsetDateTime,
        ) -> Result<AgentTask> {
            self.inner
                .repark_after_steering(parent_id, worker, lease, payload, reattach, now)
                .await
        }
        async fn pause_on_confirmation(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            continuation: ContinuationEnvelope,
            prepared_operation: Option<ListenExecutionContext>,
            now: OffsetDateTime,
        ) -> Result<AgentTask> {
            self.inner
                .pause_on_confirmation(
                    task_id,
                    worker,
                    lease,
                    continuation,
                    prepared_operation,
                    now,
                )
                .await
        }
        async fn spawn_tool_children(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            specs: Vec<ChildSpawnSpec>,
            payload: SuspensionPayload,
            child_otel_traceparent: Option<String>,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Vec<AgentTask>)> {
            self.inner
                .spawn_tool_children(
                    parent_id,
                    worker,
                    lease,
                    specs,
                    payload,
                    child_otel_traceparent,
                    now,
                )
                .await
        }
        async fn spawn_subagent_invocation(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            spawn: SubagentInvocationSpawn,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, AgentTask, AgentTask)> {
            self.inner
                .spawn_subagent_invocation(parent_id, worker, lease, spawn, now)
                .await
        }
        async fn spawn_subagent_batch(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            spawns: Vec<SubagentInvocationSpawn>,
            payload: SuspensionPayload,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Vec<(AgentTask, AgentTask)>)> {
            self.inner
                .spawn_subagent_batch(parent_id, worker, lease, spawns, payload, now)
                .await
        }
        async fn spawn_mixed_children(
            &self,
            parent_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            spawn: agent_server::journal::MixedChildrenSpawn,
            now: OffsetDateTime,
        ) -> Result<agent_server::journal::SpawnedMixedChildren> {
            self.inner
                .spawn_mixed_children(parent_id, worker, lease, spawn, now)
                .await
        }
        async fn find_subagent_invocation_for_child_root(
            &self,
            child_root_id: &AgentTaskId,
        ) -> Result<Option<AgentTask>> {
            self.inner
                .find_subagent_invocation_for_child_root(child_root_id)
                .await
        }
        async fn list_parked_subagent_invocations(&self) -> Result<Vec<AgentTask>> {
            self.inner.list_parked_subagent_invocations().await
        }
        async fn complete_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.complete_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.complete_task(task_id, worker, lease, now).await
        }
        async fn complete_task_with_result(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            result: serde_json::Value,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.complete_with_result_calls
                .fetch_add(1, Ordering::SeqCst);
            self.inner
                .complete_task_with_result(task_id, worker, lease, result, now)
                .await
        }
        async fn fail_task(
            &self,
            task_id: &AgentTaskId,
            worker: &WorkerId,
            lease: &LeaseId,
            error: String,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.inner
                .fail_task(task_id, worker, lease, error, now)
                .await
        }
        async fn cancel_tree(
            &self,
            root_id: &AgentTaskId,
            now: OffsetDateTime,
        ) -> Result<agent_server::journal::store::CancelTreeOutcome> {
            self.inner.cancel_tree(root_id, now).await
        }
        async fn resume_from_confirmation(
            &self,
            task_id: &AgentTaskId,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
            self.inner.resume_from_confirmation(task_id, now).await
        }
        async fn approve_confirmation_and_acquire(
            &self,
            task_id: &AgentTaskId,
            worker: WorkerId,
            lease: LeaseId,
            expires_at: OffsetDateTime,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<ListenExecutionContext>)> {
            self.inner
                .approve_confirmation_and_acquire(task_id, worker, lease, expires_at, now)
                .await
        }
        async fn reject_confirmation(
            &self,
            task_id: &AgentTaskId,
            error: String,
            now: OffsetDateTime,
        ) -> Result<(AgentTask, Option<AgentTask>)> {
            self.inner.reject_confirmation(task_id, error, now).await
        }
        async fn clear(&self) -> Result<()> {
            self.inner.clear().await
        }
    }

    /// Issue #301: `start_with_store_decorator` must run the whole
    /// daemon — host workers and gRPC transport — through the
    /// decorated task store. A tool-call turn drives both completion
    /// paths: the tool child commits via `complete_task_with_result`
    /// (the method bip intercepts for subagent rows) and the root turn
    /// via `complete_task`, and the flow still completes normally.
    #[tokio::test]
    async fn local_daemon_store_decorator_observes_task_completions() -> Result<()> {
        let lookup_tool = Tool {
            name: "lookup".into(),
            description: "Look something up".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "required": ["q"]
            }),
            display_name: "Lookup".into(),
            tier: ToolTier::Observe,
        };
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(vec![
            lookup_tool,
        ])));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![
                tool_use_response(
                    "resp_decor_1",
                    "tool_call_decor",
                    "lookup",
                    json!({"q": "x"}),
                ),
                text_response("resp_decor_2", "lookup done"),
            ])),
            Arc::new(ProgressToolExecutor {
                result: ToolResult::success("lookup ok"),
                emit_progress: false,
            }),
        )?;

        let complete_calls = Arc::new(AtomicUsize::new(0));
        let complete_with_result_calls = Arc::new(AtomicUsize::new(0));
        let submit_root_turn_idempotent_calls = Arc::new(AtomicUsize::new(0));
        let decorator_complete_calls = Arc::clone(&complete_calls);
        let decorator_complete_with_result_calls = Arc::clone(&complete_with_result_calls);
        let decorator_submit_calls = Arc::clone(&submit_root_turn_idempotent_calls);
        let daemon = LocalDaemon::start_with_store_decorator(
            ServiceConfig::default(),
            registry,
            runtime,
            move |mut stores| {
                stores.task_store = Arc::new(CountingTaskStore {
                    inner: Arc::clone(&stores.task_store),
                    complete_calls: decorator_complete_calls,
                    complete_with_result_calls: decorator_complete_with_result_calls,
                    submit_root_turn_idempotent_calls: decorator_submit_calls,
                    fail_get: false,
                });
                stores
            },
        )
        .await?;

        let result = async {
            let (mut control, _events) = connect_clients(&daemon.endpoint()).await?;
            let thread_id = create_thread(&mut control, "create-decorator-thread").await?;
            let _task = submit_text_work(
                &mut control,
                "submit-decorator-turn",
                &thread_id,
                "lookup x",
            )
            .await?;

            // Root turn + tool child both reach Completed through the
            // decorated store (`wait_for_completed_tasks` polls the
            // gRPC surface, which also reads through the wrapper).
            wait_for_completed_tasks(&control, &thread_id).await?;

            assert_eq!(
                complete_with_result_calls.load(Ordering::SeqCst),
                1,
                "decorator must observe the tool child's complete_task_with_result",
            );
            assert_eq!(
                complete_calls.load(Ordering::SeqCst),
                1,
                "decorator must observe the root turn's complete_task",
            );
            // gRPC-plane guard: the completion methods above are only
            // ever called by host workers, so a regression that rebuilt
            // `GrpcTransport` from an undecorated registry would still
            // pass them. `submit_root_turn_idempotent` is called
            // exactly once per `SubmitThreadWork`, and only by the
            // gRPC transport — it proves that plane also runs through
            // the decorated handles.
            assert!(
                submit_root_turn_idempotent_calls.load(Ordering::SeqCst) >= 1,
                "decorator must observe the gRPC plane's submit_root_turn_idempotent",
            );
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    async fn caller_create_direct(
        service: &GrpcControlService,
        request_id: &str,
        thread_id: &str,
    ) -> Result<String> {
        service
            .create_thread(Request::new(pb::CreateThreadRequest {
                request_id: request_id.to_owned(),
                thread_id: thread_id.to_owned(),
            }))
            .await?
            .into_inner()
            .thread
            .and_then(|view| view.thread)
            .map(|snapshot| snapshot.thread_id)
            .context("caller-addressed create response missing thread")
    }

    async fn caller_fork_direct(
        service: &GrpcControlService,
        request_id: &str,
        source_thread_id: &str,
        destination_thread_id: &str,
    ) -> Result<String> {
        service
            .fork_thread(Request::new(pb::ForkThreadRequest {
                request_id: request_id.to_owned(),
                source_thread_id: source_thread_id.to_owned(),
                fork_after_committed_turns: 0,
                thread_id: destination_thread_id.to_owned(),
            }))
            .await?
            .into_inner()
            .thread
            .and_then(|view| view.thread)
            .map(|snapshot| snapshot.thread_id)
            .context("caller-addressed fork response missing thread")
    }

    #[test]
    fn caller_minted_store_get_or_create_is_property_idempotent() -> Result<()> {
        use agent_server::journal::thread_store::{
            InMemoryThreadStore, ThreadCreationOutcome, ThreadIdConflict, ThreadStore,
        };
        use proptest::test_runner::{Config, TestCaseError, TestRunner};

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build property-test runtime")?;
        let mut runner = TestRunner::new(Config {
            cases: 32,
            ..Config::default()
        });
        runner
            .run(&"[a-z0-9]{1,20}", |suffix| {
                let result: std::result::Result<(), String> = runtime.block_on(async {
                    let store = InMemoryThreadStore::new();
                    let create_id = ThreadId::from_string(format!("create-{suffix}"));
                    let source_id = ThreadId::from_string(format!("source-{suffix}"));
                    let now = OffsetDateTime::UNIX_EPOCH;

                    let first = store
                        .get_or_create_for_creation(&create_id, &ThreadCreation::Create, now)
                        .await
                        .map_err(|error| error.to_string())?;
                    let second = store
                        .get_or_create_for_creation(&create_id, &ThreadCreation::Create, now)
                        .await
                        .map_err(|error| error.to_string())?;
                    if first != ThreadCreationOutcome::Created
                        || second != ThreadCreationOutcome::Existing
                    {
                        return Err("double-call did not create then replay".to_owned());
                    }

                    // Concurrency is NOT probed here — a current-thread
                    // proptest runtime cannot race anything. The real
                    // races live in the multi-thread spawn tests below.
                    let conflict = store
                        .get_or_create_for_creation(
                            &create_id,
                            &ThreadCreation::Fork {
                                source_thread_id: source_id,
                                fork_after_committed_turns: 0,
                            },
                            now,
                        )
                        .await;
                    match conflict {
                        Err(error) if error.downcast_ref::<ThreadIdConflict>().is_some() => {}
                        Err(error) => return Err(error.to_string()),
                        Ok(_) => {
                            return Err(
                                "conflicting call did not return ThreadIdConflict".to_owned()
                            );
                        }
                    }

                    let rows = store.list().await.map_err(|error| error.to_string())?;
                    if rows.iter().filter(|row| row.thread_id == create_id).count() != 1 {
                        return Err("double-call created more than one row".to_owned());
                    }
                    Ok(())
                });
                result.map_err(TestCaseError::fail)
            })
            .map_err(|error| anyhow!(error.to_string()))?;
        Ok(())
    }

    /// Deterministic UUID for caller-minted test ids — the boundary now
    /// enforces UUID shape, so free-form strings are rejected.
    fn caller_uuid(tag: u32) -> String {
        format!("00000000-0000-4000-8000-{tag:012x}")
    }

    #[tokio::test]
    async fn caller_thread_id_must_be_a_uuid_and_reserved_request_ids_are_rejected() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };
        let error = service
            .create_thread(Request::new(pb::CreateThreadRequest {
                request_id: "uuid-guard-1".into(),
                thread_id: "not-a-uuid".into(),
            }))
            .await
            .expect_err("non-UUID caller thread id must be rejected");
        assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error:?}");

        let error = service
            .fork_thread(Request::new(pb::ForkThreadRequest {
                request_id: "uuid-guard-2".into(),
                source_thread_id: caller_uuid(900),
                fork_after_committed_turns: 0,
                thread_id: "also-not-a-uuid".into(),
            }))
            .await
            .expect_err("non-UUID caller fork destination must be rejected");
        assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error:?}");

        let error = service
            .create_thread(Request::new(pb::CreateThreadRequest {
                request_id: "agent-sdk:thread-creation:sneaky".into(),
                thread_id: String::new(),
            }))
            .await
            .expect_err("the reserved server-internal request_id prefix must be rejected");
        assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error:?}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn caller_minted_create_double_and_concurrent_calls_reuse_one_row() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };
        let thread_id = caller_uuid(1);

        let first = caller_create_direct(&service, "caller-create-1", &thread_id).await?;
        let second = caller_create_direct(&service, "caller-create-2", &thread_id).await?;
        assert_eq!(first, thread_id);
        assert_eq!(second, thread_id);

        // A REAL race: two spawned tasks on a multi-thread runtime,
        // released by one barrier — not tokio::join! polling both
        // futures on a single task.
        let race_id = caller_uuid(2);
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let mut handles = Vec::new();
        for request_id in ["caller-race-1", "caller-race-2"] {
            let service = service.clone();
            let barrier = Arc::clone(&barrier);
            let race_id = race_id.clone();
            handles.push(tokio::spawn(async move {
                barrier.wait().await;
                caller_create_direct(&service, request_id, &race_id).await
            }));
        }
        for handle in handles {
            assert_eq!(handle.await??, race_id);
        }

        let rows = shared.stores.thread_store.list().await?;
        assert_eq!(
            rows.iter()
                .filter(|thread| thread.thread_id.0 == race_id)
                .count(),
            1,
            "racing caller-addressed creates must insert one primary-key row",
        );
        Ok(())
    }

    #[tokio::test]
    async fn caller_minted_create_rejects_conflicting_lifecycle_without_mutation() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };
        let thread_id = caller_uuid(3);
        let source_id = caller_uuid(4);
        let source = caller_create_direct(&service, "conflict-source", &source_id).await?;
        caller_create_direct(&service, "conflict-create", &thread_id).await?;
        let before = shared
            .stores
            .thread_store
            .get(&ThreadId::from_string(thread_id.clone()))
            .await?
            .context("created destination missing before conflict")?;

        let conflict = service
            .fork_thread(Request::new(pb::ForkThreadRequest {
                request_id: "conflicting-fork".to_owned(),
                source_thread_id: source,
                fork_after_committed_turns: 0,
                thread_id: thread_id.clone(),
            }))
            .await
            .expect_err("fork parameters must conflict with an existing create");
        assert_eq!(conflict.code(), tonic::Code::AlreadyExists, "{conflict:?}");
        let after = shared
            .stores
            .thread_store
            .get(&ThreadId::from_string(thread_id))
            .await?
            .context("created destination missing after conflict")?;
        assert_eq!(after, before, "conflicting call must not mutate the thread");
        Ok(())
    }

    #[tokio::test]
    async fn caller_minted_fork_replays_and_conflicts_without_mutation() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };
        let source_a = caller_create_direct(&service, "source-a-request", &caller_uuid(5)).await?;
        let source_b = caller_create_direct(&service, "source-b-request", &caller_uuid(6)).await?;
        let destination = caller_uuid(7);

        let first =
            caller_fork_direct(&service, "caller-fork-request-1", &source_a, &destination).await?;
        let retry =
            caller_fork_direct(&service, "caller-fork-request-2", &source_a, &destination).await?;
        assert_eq!(first, destination);
        assert_eq!(retry, destination);

        // Same request_id + same parameters replays the recorded result.
        let replay =
            caller_fork_direct(&service, "caller-fork-request-1", &source_a, &destination).await?;
        assert_eq!(replay, destination);

        // The request_id stays a true SECONDARY idempotency key on the
        // caller-minted path: reusing it for a DIFFERENT destination is
        // a conflict, never a second mint.
        let reused = service
            .fork_thread(Request::new(pb::ForkThreadRequest {
                request_id: "caller-fork-request-1".to_owned(),
                source_thread_id: source_a.clone(),
                fork_after_committed_turns: 0,
                thread_id: caller_uuid(8),
            }))
            .await
            .expect_err("one request_id must never mint two destinations");
        assert_eq!(reused.code(), tonic::Code::AlreadyExists, "{reused:?}");
        assert!(
            shared
                .stores
                .thread_store
                .get(&ThreadId::from_string(caller_uuid(8)))
                .await?
                .is_none(),
            "the rejected request_id reuse must not create the second destination",
        );

        let conflict = service
            .fork_thread(Request::new(pb::ForkThreadRequest {
                request_id: "caller-fork-conflict".to_owned(),
                source_thread_id: source_b,
                fork_after_committed_turns: 0,
                thread_id: destination.clone(),
            }))
            .await
            .expect_err("conflicting source must reject the caller-minted id");
        assert_eq!(conflict.code(), tonic::Code::AlreadyExists, "{conflict:?}");

        let rows = shared.stores.thread_store.list().await?;
        let destination_rows: Vec<_> = rows
            .iter()
            .filter(|thread| thread.thread_id.0 == destination)
            .collect();
        assert_eq!(destination_rows.len(), 1);
        assert_eq!(destination_rows[0].committed_turns, 0);
        assert!(
            shared
                .stores
                .message_store
                .get_history(&ThreadId::from_string(destination))
                .await?
                .is_empty(),
            "conflicting retry must not mutate the existing destination",
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn caller_minted_fork_concurrent_calls_create_once() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };
        let source_id = caller_uuid(9);
        let source = caller_create_direct(&service, "fork-race-source", &source_id).await?;
        let destination = caller_uuid(10);

        // A REAL race: spawned tasks released together by a barrier.
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let mut handles = Vec::new();
        for request_id in ["fork-race-1", "fork-race-2"] {
            let service = service.clone();
            let barrier = Arc::clone(&barrier);
            let source = source.clone();
            let destination = destination.clone();
            handles.push(tokio::spawn(async move {
                barrier.wait().await;
                caller_fork_direct(&service, request_id, &source, &destination).await
            }));
        }
        for handle in handles {
            assert_eq!(handle.await??, destination);
        }
        let rows = shared.stores.thread_store.list().await?;
        assert_eq!(
            rows.iter()
                .filter(|thread| thread.thread_id.0 == destination)
                .count(),
            1,
        );
        Ok(())
    }

    // ── ForkThread ─────────────────────────────────────────────────

    async fn fork_thread(
        control: &mut ControlClient,
        request_id: &str,
        source_thread_id: &str,
        fork_after_committed_turns: u32,
    ) -> Result<pb::ForkThreadResponse> {
        let response = control
            .fork_thread(pb::ForkThreadRequest {
                request_id: request_id.into(),
                source_thread_id: source_thread_id.into(),
                fork_after_committed_turns,
                thread_id: String::new(),
            })
            .await
            .context("fork_thread rpc")?
            .into_inner();
        Ok(response)
    }

    /// AC1: forking an unknown source thread surfaces `NOT_FOUND`
    /// without minting any state on the destination.
    #[tokio::test]
    async fn fork_thread_returns_not_found_for_unknown_source() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response(
                "resp_unused",
                "unused",
            )])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, _events) = connect_clients(&daemon.endpoint()).await?;
            let err = control
                .fork_thread(pb::ForkThreadRequest {
                    request_id: "fork-not-found".into(),
                    source_thread_id: "00000000-0000-7000-8000-000000000000".into(),
                    fork_after_committed_turns: 0,
                    thread_id: String::new(),
                })
                .await
                .expect_err("fork_thread on unknown source must error");
            assert_eq!(err.code(), tonic::Code::NotFound, "{err:?}");
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// AC2: a successful fork copies the source's committed history,
    /// re-commits its events, creates a thread aggregate (so
    /// `GetThread` / `GetThreadMessages` succeed), and is reachable
    /// via `StreamThreadEvents` replay.
    #[tokio::test]
    async fn fork_thread_copies_messages_and_events() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response(
                "resp_fork_src",
                "hello from source",
            )])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;

            // Build a source thread with one committed turn.
            let source = create_thread(&mut control, "fork-src-create").await?;
            submit_text_work(&mut control, "fork-src-submit", &source, "hi source").await?;
            let mut stream =
                open_stream(&mut events, &source, None, pb::FollowMode::ReplayAndFollow).await?;
            let _ = collect_until_closed(&mut stream).await?;

            // Snapshot the source for comparison.
            let source_messages = control
                .get_thread_messages(pb::GetThreadMessagesRequest {
                    thread_id: source.clone(),
                })
                .await
                .context("get_thread_messages source")?
                .into_inner()
                .projection
                .context("source projection missing")?;
            assert_eq!(source_messages.messages.len(), 2);

            // Fork it.
            let fork_response = fork_thread(&mut control, "fork-once", &source, 1).await?;
            let fork_thread_view = fork_response
                .thread
                .as_ref()
                .context("fork response missing thread view")?
                .thread
                .as_ref()
                .context("fork response missing snapshot")?;
            let fork_id = fork_thread_view.thread_id.clone();
            assert_ne!(fork_id, source, "fork must mint a new thread id");
            assert_eq!(fork_response.source_thread_id, source);
            assert_eq!(fork_response.message_count, 2);
            assert!(
                fork_response.event_count >= 4,
                "fork event_count should mirror source's committed events; got {}",
                fork_response.event_count,
            );

            // GetThreadMessages on the fork must succeed (regression
            // for the old bip-side fork that forgot to bootstrap the
            // thread aggregate, leaving GetThreadMessages NOT_FOUND).
            let fork_messages = control
                .get_thread_messages(pb::GetThreadMessagesRequest {
                    thread_id: fork_id.clone(),
                })
                .await
                .context("get_thread_messages fork")?
                .into_inner()
                .projection
                .context("fork projection missing")?;
            assert_eq!(fork_messages.messages.len(), source_messages.messages.len());
            for (left, right) in fork_messages
                .messages
                .iter()
                .zip(source_messages.messages.iter())
            {
                assert_eq!(left.role, right.role);
            }

            // The fork's event log must be replay-able natively —
            // this is what makes the desktop's "show prior history"
            // work without any per-host event mirror.
            let mut fork_stream =
                open_stream(&mut events, &fork_id, None, pb::FollowMode::ReplayOnly).await?;
            let fork_items = collect_until_closed(&mut fork_stream).await?;
            let fork_event_count = fork_items
                .iter()
                .filter(|item| matches!(item.item.as_ref(), Some(StreamItem::Event(_))))
                .count() as u64;
            assert_eq!(
                fork_event_count, fork_response.event_count,
                "stream replay must surface every event the fork response promised",
            );

            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// AC3: fork is idempotent under the same `request_id`. A retry
    /// returns the originally-minted thread; mismatched
    /// `source_thread_id` under the same `request_id` is rejected.
    #[tokio::test]
    async fn fork_thread_is_idempotent_under_same_request_id() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response(
                "resp_idem",
                "idempotent fork test",
            )])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;

            let source_a = create_thread(&mut control, "fork-idem-src-a").await?;
            submit_text_work(&mut control, "fork-idem-submit-a", &source_a, "first").await?;
            let mut stream = open_stream(
                &mut events,
                &source_a,
                None,
                pb::FollowMode::ReplayAndFollow,
            )
            .await?;
            let _ = collect_until_closed(&mut stream).await?;

            // First fork mints a thread.
            let first = fork_thread(&mut control, "fork-idem-key", &source_a, 1).await?;
            let first_id = first
                .thread
                .as_ref()
                .and_then(|view| view.thread.as_ref())
                .map(|snapshot| snapshot.thread_id.clone())
                .context("first fork missing thread id")?;

            // A retry under the same request_id returns the same id —
            // not a fresh mint.
            let retry = fork_thread(&mut control, "fork-idem-key", &source_a, 1).await?;
            let retry_id = retry
                .thread
                .as_ref()
                .and_then(|view| view.thread.as_ref())
                .map(|snapshot| snapshot.thread_id.clone())
                .context("retry fork missing thread id")?;
            assert_eq!(
                retry_id, first_id,
                "idempotency replay must reuse thread id"
            );

            // Same request_id with a different source is a contract
            // violation — reject loudly.
            let source_b = create_thread(&mut control, "fork-idem-src-b").await?;
            let conflict = control
                .fork_thread(pb::ForkThreadRequest {
                    request_id: "fork-idem-key".into(),
                    source_thread_id: source_b.clone(),
                    fork_after_committed_turns: 1,
                    thread_id: String::new(),
                })
                .await
                .expect_err("mismatched source under same request_id must error");
            assert_eq!(conflict.code(), tonic::Code::AlreadyExists, "{conflict:?}");

            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// AC4: source and fork diverge independently — appending work
    /// to the fork doesn't mutate the source's committed projection.
    #[tokio::test]
    async fn fork_thread_diverges_independently() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![
                text_response("resp_div_1", "first turn reply"),
                text_response("resp_div_2", "fork-only reply"),
            ])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;

            let source = create_thread(&mut control, "fork-div-create").await?;
            submit_text_work(&mut control, "fork-div-submit", &source, "first").await?;
            let mut stream =
                open_stream(&mut events, &source, None, pb::FollowMode::ReplayAndFollow).await?;
            let _ = collect_until_closed(&mut stream).await?;

            let fork_response = fork_thread(&mut control, "fork-div-fork", &source, 1).await?;
            let fork_id = fork_response
                .thread
                .as_ref()
                .and_then(|view| view.thread.as_ref())
                .map(|snapshot| snapshot.thread_id.clone())
                .context("fork response missing thread id")?;

            // Append a second turn ONLY to the fork. Wait for the
            // commit by polling `get_thread_messages` rather than
            // blocking on the event stream — `ReplayAndFollow` keeps
            // the stream open across turns, so there's no terminal
            // control frame we could anchor on.
            submit_text_work(
                &mut control,
                "fork-div-fork-submit",
                &fork_id,
                "fork-specific second turn",
            )
            .await?;
            let fork_id_for_poll = fork_id.clone();
            let _ = wait_for(|| {
                let mut control = control.clone();
                let fork_id = fork_id_for_poll.clone();
                async move {
                    let projection = control
                        .get_thread_messages(pb::GetThreadMessagesRequest { thread_id: fork_id })
                        .await
                        .context("poll get_thread_messages on fork")?
                        .into_inner()
                        .projection
                        .context("fork projection missing")?;
                    if projection.messages.len() >= 4 {
                        Ok(Some(projection))
                    } else {
                        Ok(None)
                    }
                }
            })
            .await?;

            // Source must still report just the original turn.
            let source_messages = control
                .get_thread_messages(pb::GetThreadMessagesRequest {
                    thread_id: source.clone(),
                })
                .await
                .context("get_thread_messages source")?
                .into_inner()
                .projection
                .context("source projection missing")?;
            assert_eq!(
                source_messages.messages.len(),
                2,
                "appending to the fork must not extend the source's history",
            );

            // Fork has both turns.
            let fork_messages = control
                .get_thread_messages(pb::GetThreadMessagesRequest {
                    thread_id: fork_id.clone(),
                })
                .await
                .context("get_thread_messages fork")?
                .into_inner()
                .projection
                .context("fork projection missing")?;
            assert_eq!(
                fork_messages.messages.len(),
                4,
                "fork must carry the original 2 messages plus its own divergent turn",
            );

            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// AC5: partial fork — given a 2-turn source, forking at
    /// `committed_turns=1` produces a thread carrying only the first
    /// turn's state. The second turn's events / messages are
    /// discarded. This is the load-bearing case: the desktop UX
    /// "branch from this user message" computes
    /// `fork_after_committed_turns` as the count of completed turns
    /// strictly before the chosen user message.
    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn fork_thread_partial_keeps_only_chosen_turns() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![
                text_response("resp_partial_1", "first turn reply"),
                text_response("resp_partial_2", "second turn reply"),
            ])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;

            // Build a 2-turn source.
            let source = create_thread(&mut control, "fork-partial-create").await?;
            submit_text_work(
                &mut control,
                "fork-partial-submit-1",
                &source,
                "first turn user prompt",
            )
            .await?;
            let mut stream =
                open_stream(&mut events, &source, None, pb::FollowMode::ReplayAndFollow).await?;
            let _ = collect_until_closed(&mut stream).await?;
            submit_text_work(
                &mut control,
                "fork-partial-submit-2",
                &source,
                "second turn user prompt",
            )
            .await?;
            // Wait until both turns are committed.
            let source_for_poll = source.clone();
            let _ = wait_for(|| {
                let mut control = control.clone();
                let source = source_for_poll.clone();
                async move {
                    let projection = control
                        .get_thread_messages(pb::GetThreadMessagesRequest { thread_id: source })
                        .await
                        .context("poll source projection")?
                        .into_inner()
                        .projection
                        .context("source projection missing")?;
                    if projection.messages.len() >= 4 {
                        Ok(Some(projection))
                    } else {
                        Ok(None)
                    }
                }
            })
            .await?;

            // Sanity: source has both turns.
            let source_messages = control
                .get_thread_messages(pb::GetThreadMessagesRequest {
                    thread_id: source.clone(),
                })
                .await?
                .into_inner()
                .projection
                .context("source projection missing")?;
            assert_eq!(source_messages.messages.len(), 4);

            // Fork at turn 1 — only the first turn carries over.
            let fork_response = fork_thread(&mut control, "fork-partial", &source, 1).await?;
            assert_eq!(fork_response.fork_after_committed_turns, 1);
            assert_eq!(
                fork_response.message_count, 2,
                "fork at turn 1 must carry exactly the first turn's 2 messages, not the source's 4"
            );

            let fork_id = fork_response
                .thread
                .as_ref()
                .and_then(|view| view.thread.as_ref())
                .map(|snapshot| snapshot.thread_id.clone())
                .context("fork response missing thread id")?;

            // The fork's projection has only the first turn's
            // messages — second turn dropped.
            let fork_messages = control
                .get_thread_messages(pb::GetThreadMessagesRequest {
                    thread_id: fork_id.clone(),
                })
                .await?
                .into_inner()
                .projection
                .context("fork projection missing")?;
            assert_eq!(fork_messages.messages.len(), 2);
            assert_eq!(fork_messages.messages[0].role, pb::MessageRole::User as i32);
            assert_eq!(
                fork_messages.messages[1].role,
                pb::MessageRole::Assistant as i32
            );

            // The fork's event log replays only events from the
            // first turn. The second turn's `Done` (and everything
            // tied to it) must NOT appear.
            let mut fork_stream =
                open_stream(&mut events, &fork_id, None, pb::FollowMode::ReplayOnly).await?;
            let fork_items = collect_until_closed(&mut fork_stream).await?;
            let mut done_total_turns = Vec::new();
            for item in &fork_items {
                if let Some(StreamItem::Event(event)) = item.item.as_ref()
                    && let Some(EventPayload::Done(done)) = event.event.as_ref()
                {
                    done_total_turns.push(done.total_turns);
                }
            }
            assert_eq!(
                done_total_turns,
                vec![1],
                "fork at turn 1 must replay exactly one Done event with total_turns=1; \
                 saw {done_total_turns:?}",
            );

            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// AC6: forking at turn 0 produces a brand-new thread with no
    /// history. Same shape as `CreateThread`. Useful when the user
    /// "forks" from the very first user message — i.e., wants to
    /// start a parallel conversation from scratch but keep the
    /// runtime config the source had.
    #[tokio::test]
    async fn fork_thread_at_turn_zero_yields_empty_thread() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response(
                "resp_zero",
                "ignored",
            )])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;

            let source = create_thread(&mut control, "fork-zero-create").await?;
            submit_text_work(&mut control, "fork-zero-submit", &source, "noise").await?;
            let mut stream =
                open_stream(&mut events, &source, None, pb::FollowMode::ReplayAndFollow).await?;
            let _ = collect_until_closed(&mut stream).await?;

            // Fork at turn 0.
            let fork_response = fork_thread(&mut control, "fork-zero", &source, 0).await?;
            assert_eq!(fork_response.fork_after_committed_turns, 0);
            assert_eq!(fork_response.message_count, 0);
            assert_eq!(fork_response.event_count, 0);

            let fork_id = fork_response
                .thread
                .as_ref()
                .and_then(|view| view.thread.as_ref())
                .map(|snapshot| snapshot.thread_id.clone())
                .context("fork response missing thread id")?;

            let fork_messages = control
                .get_thread_messages(pb::GetThreadMessagesRequest {
                    thread_id: fork_id.clone(),
                })
                .await?
                .into_inner()
                .projection
                .context("fork projection missing")?;
            assert!(fork_messages.messages.is_empty());
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// AC7: a `fork_after_committed_turns` greater than the source's
    /// `committed_turns` is `INVALID_ARGUMENT`. Catches off-by-one
    /// bugs in callers that compute the boundary from message
    /// indices.
    #[tokio::test]
    async fn fork_thread_rejects_out_of_range_turn() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response(
                "resp_oor",
                "single turn",
            )])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, mut events) = connect_clients(&daemon.endpoint()).await?;
            let source = create_thread(&mut control, "fork-oor-create").await?;
            submit_text_work(&mut control, "fork-oor-submit", &source, "only turn").await?;
            let mut stream =
                open_stream(&mut events, &source, None, pb::FollowMode::ReplayAndFollow).await?;
            let _ = collect_until_closed(&mut stream).await?;

            // Source has committed_turns == 1; ask for turn 2.
            let err = control
                .fork_thread(pb::ForkThreadRequest {
                    request_id: "fork-oor".into(),
                    source_thread_id: source,
                    fork_after_committed_turns: 2,
                    thread_id: String::new(),
                })
                .await
                .expect_err("forking past committed_turns must reject");
            assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err:?}");
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    // ── Phase 10 · E: idempotency + back-pressure + input limits ─────

    /// A retried `SubmitThreadWork` under the same `request_id` returns
    /// the originally-admitted task (durable dedup) and surfaces the
    /// queue depth in the response.
    #[tokio::test]
    async fn submit_thread_work_is_idempotent_under_same_request_id() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response("r", "ok")])),
            Arc::new(NoopToolExecutor),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, _events) = connect_clients(&daemon.endpoint()).await?;
            let thread = create_thread(&mut control, "idem-create").await?;

            let first = control
                .submit_thread_work(pb::SubmitThreadWorkRequest {
                    request_id: "idem-submit".into(),
                    thread_id: thread.clone(),
                    input: vec![text_input("hello")],
                    caller_metadata: String::new(),
                })
                .await
                .context("first submit")?
                .into_inner();
            let first_task = first.task.context("first task")?;

            // A retry with the same request_id + payload returns the
            // same task id without admitting a second root.
            let retry = control
                .submit_thread_work(pb::SubmitThreadWorkRequest {
                    request_id: "idem-submit".into(),
                    thread_id: thread.clone(),
                    input: vec![text_input("hello")],
                    caller_metadata: String::new(),
                })
                .await
                .context("retry submit")?
                .into_inner();
            let retry_task = retry.task.context("retry task")?;
            assert_eq!(
                retry_task.task_id, first_task.task_id,
                "retry replays original"
            );

            // A retry with a *different* payload under the same key is a
            // conflict, not a silent alias.
            let conflict = control
                .submit_thread_work(pb::SubmitThreadWorkRequest {
                    request_id: "idem-submit".into(),
                    thread_id: thread.clone(),
                    input: vec![text_input("different")],
                    caller_metadata: String::new(),
                })
                .await
                .expect_err("payload change under same key must conflict");
            assert_eq!(conflict.code(), tonic::Code::AlreadyExists, "{conflict:?}");
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// Oversized input returns a clean `INVALID_ARGUMENT`, not a
    /// transport failure or OOM.
    #[tokio::test]
    async fn submit_thread_work_rejects_oversized_input() -> Result<()> {
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
            Vec::new(),
        )));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![text_response("r", "ok")])),
            Arc::new(NoopToolExecutor),
        )?;
        // Tighten the per-item cap so the test payload stays small.
        let config = ServiceConfig {
            admission: crate::config::AdmissionConfig {
                max_submit_item_bytes: Some(64),
                max_submit_input_bytes: Some(128),
                ..crate::config::AdmissionConfig::default()
            },
            ..ServiceConfig::default()
        };
        let daemon = LocalDaemon::start(config, registry, runtime).await?;

        let result = async {
            let (mut control, _events) = connect_clients(&daemon.endpoint()).await?;
            let thread = create_thread(&mut control, "big-create").await?;

            let huge = "x".repeat(256);
            let err = control
                .submit_thread_work(pb::SubmitThreadWorkRequest {
                    request_id: "big-submit".into(),
                    thread_id: thread.clone(),
                    input: vec![text_input(&huge)],
                    caller_metadata: String::new(),
                })
                .await
                .expect_err("oversized input must reject");
            assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err:?}");
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// Caller metadata supplied on `SubmitThreadWork` is captured durably on
    /// the admitted root turn, so it is readable when the definition resolves
    /// per-turn tools (`AgentDefinition::resolve_tools`) at turn start. An
    /// empty payload leaves the task's `caller_metadata` `None` (identical to
    /// pre-field behavior), and a malformed JSON payload is rejected up front
    /// with `INVALID_ARGUMENT` before any durable work.
    #[tokio::test]
    async fn submit_thread_work_captures_caller_metadata_on_task() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };

        let thread = service
            .create_thread(Request::new(pb::CreateThreadRequest {
                request_id: "meta-create".into(),
                thread_id: String::new(),
            }))
            .await?
            .into_inner()
            .thread
            .and_then(|view| view.thread)
            .context("create_thread missing snapshot")?
            .thread_id;

        // Present + valid → parsed and captured on the task verbatim.
        let armed = service
            .submit_thread_work(Request::new(pb::SubmitThreadWorkRequest {
                request_id: "meta-armed".into(),
                thread_id: thread.clone(),
                input: vec![text_input("armed turn")],
                caller_metadata: r#"{"plan_mode":true}"#.into(),
            }))
            .await?
            .into_inner()
            .task
            .context("submit missing task")?;
        let armed_task = shared
            .stores
            .task_store
            .get(&AgentTaskId::from_string(armed.task_id.clone()))
            .await?
            .context("armed task not persisted")?;
        assert_eq!(
            armed_task.caller_metadata,
            Some(serde_json::json!({ "plan_mode": true })),
            "caller_metadata must round-trip onto the durable task"
        );

        // Absent (empty string) → task carries `None`, exactly as before the
        // field existed; the definition's static tool list is used.
        let plain = service
            .submit_thread_work(Request::new(pb::SubmitThreadWorkRequest {
                request_id: "meta-plain".into(),
                thread_id: thread.clone(),
                input: vec![text_input("plain turn")],
                caller_metadata: String::new(),
            }))
            .await?
            .into_inner()
            .task
            .context("submit missing task")?;
        let plain_task = shared
            .stores
            .task_store
            .get(&AgentTaskId::from_string(plain.task_id.clone()))
            .await?
            .context("plain task not persisted")?;
        assert_eq!(plain_task.caller_metadata, None);

        // Malformed JSON → rejected before admission (no durable task).
        let err = service
            .submit_thread_work(Request::new(pb::SubmitThreadWorkRequest {
                request_id: "meta-bad".into(),
                thread_id: thread.clone(),
                input: vec![text_input("bad turn")],
                caller_metadata: "{not json".into(),
            }))
            .await
            .expect_err("malformed caller_metadata must reject");
        assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err:?}");

        Ok(())
    }

    /// Create a thread and admit one root turn through the in-process
    /// control service. No host/worker runs in this fixture, so the
    /// returned root turn stays non-terminal (`Pending`) until a cancel
    /// transitions it — exactly the state `CancelTask` must handle.
    ///
    /// Returns `(thread_id, root_task_id)`.
    async fn seed_pending_root(
        service: &GrpcControlService,
        label: &str,
    ) -> Result<(String, String)> {
        let thread = service
            .create_thread(Request::new(pb::CreateThreadRequest {
                request_id: format!("{label}-create"),
                thread_id: String::new(),
            }))
            .await?
            .into_inner()
            .thread
            .and_then(|view| view.thread)
            .context("create_thread missing snapshot")?
            .thread_id;
        let task = service
            .submit_thread_work(Request::new(pb::SubmitThreadWorkRequest {
                request_id: format!("{label}-submit"),
                thread_id: thread.clone(),
                input: vec![text_input("do work")],
                caller_metadata: String::new(),
            }))
            .await?
            .into_inner()
            .task
            .context("submit_thread_work missing task")?;
        Ok((thread, task.task_id))
    }

    /// `CancelTask` transitions a queued/running root turn to
    /// `CANCELLED` and reports the ids it cancelled.
    #[tokio::test]
    async fn cancel_task_cancels_a_pending_root_turn() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };

        let (_thread, task_id) = seed_pending_root(&service, "cancel-pending").await?;

        let cancel = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: task_id.clone(),
                reason: "user aborted".into(),
                thread_id: String::new(),
            }))
            .await?
            .into_inner();
        assert_eq!(cancel.cancelled_task_ids, vec![task_id.clone()]);
        assert_eq!(cancel.cancelled_count, 1);

        let after = service
            .get_task(Request::new(pb::GetTaskRequest {
                task_id: task_id.clone(),
            }))
            .await?
            .into_inner()
            .task
            .context("get_task missing task")?;
        assert_eq!(after.status, pb::TaskStatus::Cancelled as i32, "{after:?}");
        Ok(())
    }

    /// Cancelling an unknown task id is a clean `NOT_FOUND`, not the
    /// INTERNAL that `cancel_tree`'s own existence check would surface.
    #[tokio::test]
    async fn cancel_task_returns_not_found_for_unknown_task() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService { shared };

        let err = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: "00000000-0000-7000-8000-000000000000".into(),
                reason: String::new(),
                thread_id: String::new(),
            }))
            .await
            .expect_err("cancel on unknown task must error");
        assert_eq!(err.code(), tonic::Code::NotFound, "{err:?}");
        Ok(())
    }

    /// Tool executor whose side effect lands only after `delay` — and
    /// whose cancel argument aborts it first, like any well-behaved
    /// external tool. `started` observes dispatch; `landed` observes
    /// the side effect.
    struct SlowSideEffectExecutor {
        started: Arc<std::sync::atomic::AtomicBool>,
        landed: Arc<std::sync::atomic::AtomicBool>,
        delay: Duration,
    }

    #[async_trait]
    impl ToolCallExecutor for SlowSideEffectExecutor {
        async fn execute_tool_call(
            &self,
            _bootstrap: &agent_server::ToolTaskBootstrap,
            _collector: agent_server::worker::ToolEventCollector,
            cancel: CancellationToken,
        ) -> Result<ToolResult> {
            self.started
                .store(true, std::sync::atomic::Ordering::SeqCst);
            tokio::select! {
                () = cancel.cancelled() => Ok(ToolResult {
                    success: false,
                    output: "aborted before side effect".into(),
                    data: None,
                    documents: Vec::new(),
                    duration_ms: None,
                }),
                () = tokio::time::sleep(self.delay) => {
                    self.landed
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    Ok(ToolResult::success("side effect landed"))
                }
            }
        }
    }

    /// Issue #299 round 4 (item 1): an APPROVED Confirm-tier tool runs
    /// on a detached drive under the server shutdown token; before
    /// this fix, cancelling its task row (`CancelTask`, the subagent
    /// deadline sweep) left the drive executing and its side effect
    /// could land AFTER the cancellation. The drive now registers a
    /// per-drive token that the cancel paths trip, and the executor's
    /// cancel argument observes it before the effect lands.
    #[tokio::test]
    async fn cancel_task_aborts_in_flight_approved_confirm_drive() -> Result<()> {
        let transfer_tool = Tool {
            name: "transfer".into(),
            description: "Transfer funds".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "amount": { "type": "number" } },
                "required": ["amount"]
            }),
            display_name: "Transfer".into(),
            tier: ToolTier::Confirm,
        };
        let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(vec![
            transfer_tool,
        ])));
        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let landed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let runtime = runtime_with(
            Arc::new(ScriptedProvider::new(vec![tool_use_response(
                "resp_drive_cancel_1",
                "tool_call_drive_cancel",
                "transfer",
                json!({"amount": 7}),
            )])),
            Arc::new(SlowSideEffectExecutor {
                started: Arc::clone(&started),
                landed: Arc::clone(&landed),
                delay: Duration::from_millis(1_500),
            }),
        )?;
        let daemon = LocalDaemon::start(ServiceConfig::default(), registry, runtime).await?;

        let result = async {
            let (mut control, _events) = connect_clients(&daemon.endpoint()).await?;
            let thread_id = create_thread(&mut control, "drive-cancel-create").await?;
            let root = submit_text_work(
                &mut control,
                "drive-cancel-submit",
                &thread_id,
                "transfer 7",
            )
            .await?;

            let awaiting = wait_for_awaiting_confirmation(&control, &thread_id).await?;
            control
                .decide_confirmation(pb::DecideConfirmationRequest {
                    request_id: "drive-cancel-approve".into(),
                    thread_id: thread_id.clone(),
                    task_id: awaiting.task_id.clone(),
                    decision: Some(pb::ConfirmationDecision {
                        decision: Some(pb::confirmation_decision::Decision::Approved(
                            pb::ApprovedConfirmation {},
                        )),
                    }),
                })
                .await
                .context("decide_confirmation rpc")?;

            // Wait until the tool is genuinely mid-flight on the
            // detached drive, then cancel the root task tree.
            let started_probe = Arc::clone(&started);
            wait_for(move || {
                let started_probe = Arc::clone(&started_probe);
                async move {
                    Ok(started_probe
                        .load(std::sync::atomic::Ordering::SeqCst)
                        .then_some(()))
                }
            })
            .await
            .context("tool never started")?;

            let cancelled = control
                .cancel_task(pb::CancelTaskRequest {
                    task_id: root.task_id.clone(),
                    reason: "deadline enforcement".into(),
                    thread_id: String::new(),
                })
                .await
                .context("cancel_task rpc")?
                .into_inner();
            assert!(
                cancelled
                    .cancelled_task_ids
                    .contains(&awaiting.task_id.clone()),
                "the approved tool task must be in the cancelled set, got {cancelled:?}",
            );

            // Give the tool's 1.5s side-effect timer ample time to
            // have fired if the cancel had NOT reached the drive.
            tokio::time::sleep(Duration::from_millis(2_500)).await;
            assert!(
                !landed.load(std::sync::atomic::Ordering::SeqCst),
                "the tool's side effect must not land after its drive was cancelled",
            );

            let tasks = list_thread_tasks(&mut control, &thread_id, "post drive cancel").await?;
            let tool_row = tasks
                .iter()
                .find(|task| task.task_id == awaiting.task_id)
                .context("tool task listed")?;
            assert_eq!(
                tool_row.status,
                pb::TaskStatus::Cancelled as i32,
                "the approved tool task must end Cancelled, got {tool_row:?}",
            );
            Ok(())
        }
        .await;

        daemon.stop().await?;
        result
    }

    /// A blank `task_id` is rejected up front with `INVALID_ARGUMENT`.
    #[tokio::test]
    async fn cancel_task_rejects_blank_task_id() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService { shared };

        let err = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: "   ".into(),
                reason: String::new(),
                thread_id: String::new(),
            }))
            .await
            .expect_err("blank task_id must reject");
        assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err:?}");
        Ok(())
    }

    /// Re-cancelling an already-terminal subtree is a no-op: the second
    /// call succeeds and reports zero newly-cancelled ids.
    #[tokio::test]
    async fn cancel_task_is_idempotent_on_already_terminal_subtree() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };

        let (_thread, task_id) = seed_pending_root(&service, "cancel-idem").await?;

        let first = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: task_id.clone(),
                reason: String::new(),
                thread_id: String::new(),
            }))
            .await?
            .into_inner();
        assert_eq!(first.cancelled_count, 1);

        let second = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: task_id.clone(),
                reason: String::new(),
                thread_id: String::new(),
            }))
            .await?
            .into_inner();
        assert!(
            second.cancelled_task_ids.is_empty(),
            "{:?}",
            second.cancelled_task_ids,
        );
        assert_eq!(second.cancelled_count, 0);
        Ok(())
    }

    /// Seed a real interior task: admit a root turn, lease it `Running`,
    /// then spawn one `ToolRuntime` child under it. The child is a
    /// non-root, out-of-subtree task — the case `CancelTask` must reject.
    ///
    /// Returns `(thread_id, root_task_id, child_task_id)`.
    async fn seed_interior_tool_child(
        shared: &GrpcShared,
        thread_name: &str,
    ) -> Result<(ThreadId, String, String)> {
        let now = OffsetDateTime::UNIX_EPOCH;
        let thread_id = ThreadId::from_string(thread_name);
        let root = AgentTask::new_root_turn(thread_id.clone(), now, 5);
        let root_id = root.id.clone();
        let store = &shared.stores.task_store;
        store
            .submit_root_turn(root.clone())
            .await
            .context("submit root for interior-child fixture")?;

        let worker = WorkerId::from_string("w-interior");
        let lease = LeaseId::from_string("l-interior");
        store
            .try_acquire_task(
                &root_id,
                worker.clone(),
                lease.clone(),
                now + time::Duration::seconds(60),
                now,
            )
            .await
            .context("acquire root for interior-child fixture")?
            .context("acquire returned None")?;

        let continuation = ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread_id.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: Vec::new(),
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread_id.clone()),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        });
        let (_parent, children) = store
            .spawn_tool_children(
                &root_id,
                &worker,
                &lease,
                vec![ChildSpawnSpec::new(2)],
                SuspensionPayload {
                    continuation,
                    suspended_messages: Vec::new(),
                },
                None,
                now,
            )
            .await
            .context("spawn tool child for interior-child fixture")?;
        let child = children
            .first()
            .context("spawn_tool_children returned no child")?;
        assert!(!child.kind.is_root(), "child must be a non-root task");

        Ok((thread_id, root_id.to_string(), child.id.to_string()))
    }

    /// `CancelTask` rejects an interior (non-root) task id with
    /// `FAILED_PRECONDITION`. Cancelling such a task on the durable SQL
    /// backends would strand the out-of-subtree parent in
    /// `WaitingOnChildren`, so the RPC only supports root tasks.
    #[tokio::test]
    async fn cancel_task_rejects_interior_task() -> Result<()> {
        let shared = event_test_shared()?;
        let (_thread, _root_id, child_id) =
            seed_interior_tool_child(&shared, "t-cancel-interior").await?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };

        let Err(err) = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: child_id,
                reason: String::new(),
                thread_id: String::new(),
            }))
            .await
        else {
            bail!("cancelling an interior task must reject");
        };
        assert_eq!(err.code(), tonic::Code::FailedPrecondition, "{err:?}");
        Ok(())
    }

    /// The headline behavior through the RPC: cancelling a root with a
    /// live descendant cancels the whole subtree and reports the ids in
    /// cancellation order (root first, then the child).
    #[tokio::test]
    async fn cancel_task_cascades_to_descendants() -> Result<()> {
        let shared = event_test_shared()?;
        let (_thread, root_id, child_id) =
            seed_interior_tool_child(&shared, "t-cancel-cascade").await?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };

        let cancel = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: root_id.clone(),
                reason: "cascade".into(),
                thread_id: String::new(),
            }))
            .await?
            .into_inner();
        assert_eq!(cancel.cancelled_task_ids, vec![root_id, child_id]);
        assert_eq!(cancel.cancelled_count, 2);
        Ok(())
    }

    /// The client-controlled cancel `reason` is bounded before it
    /// reaches structured logs: short values pass through unallocated,
    /// oversized ones are cut on a char boundary with an ellipsis.
    #[test]
    fn bounded_log_str_truncates_oversized_client_strings() {
        let short = "user aborted";
        assert!(matches!(
            bounded_log_str(short, MAX_CANCEL_REASON_LOG_CHARS),
            std::borrow::Cow::Borrowed(_)
        ));

        // Multi-byte chars: truncation must stay on a char boundary.
        let oversized = "é".repeat(MAX_CANCEL_REASON_LOG_CHARS + 100);
        let bounded = bounded_log_str(&oversized, MAX_CANCEL_REASON_LOG_CHARS);
        assert_eq!(
            bounded.chars().count(),
            MAX_CANCEL_REASON_LOG_CHARS + 1,
            "capped chars plus the ellipsis marker",
        );
        assert!(bounded.ends_with('…'));
    }

    /// Read the next frame from an in-process event stream, bounded so a
    /// missing terminal frame fails the test instead of hanging it.
    async fn next_inline_frame(stream: &mut EventStream) -> Result<pb::StreamThreadEventsResponse> {
        tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .context("timed out waiting for a stream frame")?
            .context("stream ended unexpectedly")?
            .context("stream frame error")
    }

    /// Cancelling a parked root gives an existing `REPLAY_AND_FOLLOW`
    /// subscriber a terminal `Cancelled` event frame followed by a
    /// `Closed(TURN_CANCELLED)` control frame — without it the follower
    /// would wait forever for a `Done` that never comes (a cancelled
    /// turn commits no other lifecycle event).
    #[tokio::test]
    async fn cancel_task_gives_followers_a_terminal_cancelled_frame() -> Result<()> {
        let shared = event_test_shared()?;
        let control = GrpcControlService {
            shared: Arc::clone(&shared),
        };
        let events = GrpcEventService {
            shared: Arc::clone(&shared),
        };

        // Mint a real thread row first — the event stream RPC rejects
        // unknown threads — then park a root (with a tool child) on it.
        let thread_name = control
            .create_thread(Request::new(pb::CreateThreadRequest {
                request_id: "cancel-follower-create".into(),
                thread_id: String::new(),
            }))
            .await?
            .into_inner()
            .thread
            .and_then(|view| view.thread)
            .context("create_thread missing snapshot")?
            .thread_id;
        let (thread_id, root_id, _child_id) =
            seed_interior_tool_child(&shared, &thread_name).await?;

        let mut stream = events
            .stream_thread_events(Request::new(pb::StreamThreadEventsRequest {
                thread_id: thread_id.to_string(),
                after_sequence: None,
                follow_mode: pb::FollowMode::ReplayAndFollow as i32,
            }))
            .await?
            .into_inner();

        // Drain the (empty-journal) replay phase.
        let opened = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(opened.item, Some(StreamItem::ReplayOpened(_))),
            "expected replay_opened, got {opened:?}",
        );
        let catchup = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(catchup.item, Some(StreamItem::ReplayCatchupComplete(_))),
            "expected replay_catchup_complete, got {catchup:?}",
        );

        control
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: root_id,
                reason: "follower test".into(),
                thread_id: String::new(),
            }))
            .await?;

        // The follower receives the terminal Cancelled event...
        let event_frame = next_inline_frame(&mut stream).await?;
        let Some(StreamItem::Event(envelope)) = event_frame.item else {
            bail!("expected an event frame, got {event_frame:?}");
        };
        ensure!(
            matches!(
                envelope.event,
                Some(pb::event_envelope::Event::Cancelled(_))
            ),
            "expected a cancelled event, got {:?}",
            envelope.event,
        );

        // ...then the stream closes with the cancelled reason.
        let closed_frame = next_inline_frame(&mut stream).await?;
        let Some(StreamItem::Closed(closed)) = closed_frame.item else {
            bail!("expected a closed frame, got {closed_frame:?}");
        };
        assert_eq!(
            closed.reason,
            pb::StreamCloseReason::TurnCancelled as i32,
            "{closed:?}",
        );
        Ok(())
    }

    /// Cancelling a QUEUED root behind a live active root must NOT close
    /// the thread's followers: no `Cancelled` marker is committed, the
    /// stream stays open, and subsequent events (the active root's work)
    /// are still delivered.
    #[tokio::test]
    async fn cancel_of_queued_root_leaves_followers_streaming() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;

        let shared = event_test_shared()?;
        let control = GrpcControlService {
            shared: Arc::clone(&shared),
        };
        let events = GrpcEventService {
            shared: Arc::clone(&shared),
        };

        // Active root occupies the slot; the second submit queues.
        let (thread, _active_task_id) = seed_pending_root(&control, "cancel-queued-follow").await?;
        let queued = control
            .submit_thread_work(Request::new(pb::SubmitThreadWorkRequest {
                request_id: "cancel-queued-follow-second".into(),
                thread_id: thread.clone(),
                input: vec![text_input("queued work")],
                caller_metadata: String::new(),
            }))
            .await?
            .into_inner()
            .task
            .context("second submit missing task")?;
        assert_eq!(
            queued.status,
            pb::TaskStatus::Queued as i32,
            "precondition: second root queues behind the active one",
        );

        let mut stream = events
            .stream_thread_events(Request::new(pb::StreamThreadEventsRequest {
                thread_id: thread.clone(),
                after_sequence: None,
                follow_mode: pb::FollowMode::ReplayAndFollow as i32,
            }))
            .await?
            .into_inner();
        let opened = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(opened.item, Some(StreamItem::ReplayOpened(_))),
            "expected replay_opened, got {opened:?}",
        );
        let catchup = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(catchup.item, Some(StreamItem::ReplayCatchupComplete(_))),
            "expected replay_catchup_complete, got {catchup:?}",
        );

        let cancel = control
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: queued.task_id.clone(),
                reason: "queued cancel".into(),
                thread_id: String::new(),
            }))
            .await?
            .into_inner();
        assert_eq!(cancel.cancelled_task_ids, vec![queued.task_id]);

        // Prove the follower is still live: commit + notify a probe
        // event and assert it is the very next frame — not a Cancelled
        // marker, not a Closed frame.
        let committed = shared
            .stores
            .event_repo
            .commit_event(
                &ThreadId::from_string(&thread),
                AgentEvent::TextDelta {
                    message_id: "probe".into(),
                    delta: "still streaming".into(),
                },
                OffsetDateTime::now_utc(),
            )
            .await?;
        shared
            .stores
            .event_notifier
            .notify(std::slice::from_ref(&committed));

        let frame = next_inline_frame(&mut stream).await?;
        let Some(StreamItem::Event(envelope)) = frame.item else {
            bail!("expected the probe event frame, got {frame:?}");
        };
        ensure!(
            matches!(
                envelope.event,
                Some(pb::event_envelope::Event::TextDelta(_))
            ),
            "queued cancel must not interject a Cancelled marker, got {:?}",
            envelope.event,
        );
        Ok(())
    }

    /// `CancelTask` enforces the thread guard: a `(thread_id, task_id)`
    /// pair whose task belongs to a different thread is rejected with
    /// `FAILED_PRECONDITION` rather than cancelling another thread's work.
    #[tokio::test]
    async fn cancel_task_rejects_thread_mismatch() -> Result<()> {
        let shared = event_test_shared()?;
        let service = GrpcControlService {
            shared: Arc::clone(&shared),
        };

        let (thread, task_id) = seed_pending_root(&service, "cancel-thread-guard").await?;

        let Err(err) = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: task_id.clone(),
                reason: String::new(),
                thread_id: format!("{thread}-other"),
            }))
            .await
        else {
            bail!("thread mismatch must reject");
        };
        assert_eq!(err.code(), tonic::Code::FailedPrecondition, "{err:?}");

        // The matching thread guard still permits the cancel, proving the
        // guard only blocks genuine mismatches.
        let ok = service
            .cancel_task(Request::new(pb::CancelTaskRequest {
                task_id: task_id.clone(),
                reason: String::new(),
                thread_id: thread,
            }))
            .await?
            .into_inner();
        assert_eq!(ok.cancelled_task_ids, vec![task_id]);
        Ok(())
    }

    fn budget_exceeded_event() -> AgentEvent {
        AgentEvent::BudgetExceeded {
            thread_id: ThreadId::from_string("t-budget"),
            total_turns: 3,
            total_usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            duration: Duration::from_millis(2500),
            estimated_cost_usd: Some(0.25),
            limit: agent_sdk_foundation::types::BudgetLimitKind::TotalTokens,
            emitter_task_id: None,
        }
    }

    #[test]
    fn budget_exceeded_projects_to_done_event() -> Result<()> {
        // Previously `BudgetExceeded` fell through to `Status::internal`, so a
        // budget-terminated thread's replay/follow stream failed with INTERNAL.
        // It must now project to a terminal `DoneEvent` carrying the run totals.
        let payload = map_event_payload(&budget_exceeded_event())?;
        match payload {
            EventPayload::Done(done) => {
                assert_eq!(done.thread_id, "t-budget");
                assert_eq!(done.total_turns, 3);
                let usage = done.total_usage.context("done event must carry usage")?;
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 50);
                assert_eq!(done.stop_reason(), pb::RunStopReason::BudgetTotalTokens);
                assert_eq!(done.estimated_cost_usd, Some(0.25));
                let duration = done
                    .duration
                    .context("budget stop must carry the run's wall-clock duration")?;
                assert_eq!(duration.seconds, 2);
                assert_eq!(duration.nanos, 500_000_000);
            }
            other => bail!("BudgetExceeded must map to a Done event, got {other:?}"),
        }
        Ok(())
    }

    /// Shared fixture for the terminal-close gate tests: a thread, a
    /// predecessor `Done` frame for turn 1, and a checkpoint factory.
    fn close_gate_fixture(
        name: &str,
    ) -> Result<(Arc<GrpcShared>, ThreadId, agent_server::CommittedEvent)> {
        let shared = event_test_shared()?;
        let thread = ThreadId::from_string(name);
        let done = close_gate_done(&thread, 9, 1);
        Ok((shared, thread, done))
    }

    /// An id-less `Done` frame — the shape journaled before emitter
    /// attribution existed, and the one every back-compat assertion
    /// pins.
    fn close_gate_done(
        thread: &ThreadId,
        sequence: u64,
        total_turns: usize,
    ) -> agent_server::CommittedEvent {
        agent_server::CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: thread.clone(),
            sequence,
            timestamp: OffsetDateTime::UNIX_EPOCH,
            event: AgentEvent::done(
                thread.clone(),
                total_turns,
                agent_sdk_foundation::TokenUsage::default(),
                std::time::Duration::from_secs(1),
            ),
        }
    }

    /// The same frame, attributed to the task that committed it.
    fn close_gate_done_from(
        thread: &ThreadId,
        sequence: u64,
        total_turns: usize,
        emitter: &agent_server::journal::task::AgentTaskId,
    ) -> agent_server::CommittedEvent {
        let mut committed = close_gate_done(thread, sequence, total_turns);
        committed.event = committed.event.with_emitter_task_id(emitter.as_str());
        committed
    }

    fn close_gate_checkpoint(
        thread: &ThreadId,
        task_id: &str,
        turn: u32,
    ) -> agent_server::journal::checkpoint::NewCheckpointParams {
        agent_server::journal::checkpoint::NewCheckpointParams {
            thread_id: thread.clone(),
            turn_number: turn,
            task_id: agent_server::journal::task::AgentTaskId::from_string(task_id),
            messages: vec![],
            agent_state_snapshot: serde_json::json!({}),
            turn_usage: agent_sdk_foundation::TokenUsage::default(),
            kind: agent_server::journal::checkpoint::CheckpointKind::FullTurn,
            now: OffsetDateTime::UNIX_EPOCH,
        }
    }

    /// Suppress arm: a terminal frame whose turn is behind the
    /// thread's latest committed checkpoint is stale —
    /// later events (the collision successor's boundary/answer/`Done`,
    /// or newer turns in backfill) must not be truncated. Holds
    /// whether the successor is live or already terminal: the
    /// discriminator is durable monotonic state, never task rows.
    #[tokio::test]
    async fn stale_done_behind_committed_turns_never_closes() -> Result<()> {
        let (shared, thread, done) = close_gate_fixture("t-stale-done-suppress")?;
        shared
            .stores
            .thread_store
            .get_or_create(&thread, OffsetDateTime::UNIX_EPOCH)
            .await?;
        shared
            .stores
            .checkpoint_store
            .commit_checkpoint(close_gate_checkpoint(
                &thread,
                "task_foreign_predecessor",
                1,
            ))
            .await?;

        // The successor committed its own LATER turn: the old frame is
        // stale regardless of any task state.
        let successor = agent_server::journal::task::AgentTask::new_root_turn(
            thread.clone(),
            OffsetDateTime::UNIX_EPOCH,
            3,
        );
        let successor_id = successor.id.clone();
        shared.stores.task_store.submit_root_turn(successor).await?;
        shared
            .stores
            .checkpoint_store
            .commit_checkpoint(close_gate_checkpoint(&thread, successor_id.as_str(), 2))
            .await?;
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &done).await,
            None,
            "an old foreign Done must not close after a later turn committed",
        );

        // Attribution never overrides the monotonic rule: a frame behind
        // the head stays suppressed even when its emitter is perfectly
        // healthy.
        assert_eq!(
            confirmed_close_reason(
                &shared,
                &thread,
                &close_gate_done_from(&thread, 9, 1, &successor_id)
            )
            .await,
            None,
            "a stale frame must not close just because its emitter is live",
        );

        // Still stale after the successor is TERMINAL.
        let outcome = shared
            .stores
            .task_store
            .cancel_tree(&successor_id, OffsetDateTime::UNIX_EPOCH)
            .await?;
        assert!(!outcome.transitioned.is_empty(), "successor left the slot");
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &done).await,
            None,
            "a stale foreign Done must not close even after the successor is terminal",
        );
        Ok(())
    }

    /// Close arms: a frame at the thread's latest committed turn
    /// always closes — deterministically, with no task-state reads.
    /// Pins that a queued root promoted before publish must NOT make
    /// a valid `Done` timing-dependent, and that `Cancelled` markers
    /// are never gated.
    #[tokio::test]
    async fn frame_at_committed_head_always_closes() -> Result<()> {
        let (shared, thread, done) = close_gate_fixture("t-head-done-close")?;
        shared
            .stores
            .thread_store
            .get_or_create(&thread, OffsetDateTime::UNIX_EPOCH)
            .await?;

        // No checkpoints at all: historical close.
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &done).await,
            Some(pb::StreamCloseReason::ThreadCompleted),
        );

        // The emitter committed the frame's turn; a PROMOTED queued
        // root is already the blocking occupant (normal multi-turn
        // promotion). The valid Done must still close: no task-state
        // inference.
        let emitter = agent_server::journal::task::AgentTask::new_root_turn(
            thread.clone(),
            OffsetDateTime::UNIX_EPOCH,
            3,
        );
        let emitter_id = emitter.id.clone();
        shared.stores.task_store.submit_root_turn(emitter).await?;
        shared
            .stores
            .checkpoint_store
            .commit_checkpoint(close_gate_checkpoint(&thread, emitter_id.as_str(), 1))
            .await?;
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &done).await,
            Some(pb::StreamCloseReason::ThreadCompleted),
            "a Done at the committed head closes regardless of the blocking root",
        );

        // Cancelled markers are not gated.
        let cancelled = agent_server::CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: thread.clone(),
            sequence: 10,
            timestamp: OffsetDateTime::UNIX_EPOCH,
            event: AgentEvent::cancelled(1, agent_sdk_foundation::TokenUsage::default()),
        };
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &cancelled).await,
            Some(pb::StreamCloseReason::TurnCancelled),
        );
        Ok(())
    }

    /// Submit a root turn on `thread` and commit its checkpoint at
    /// `turn`, returning the id the frames under test are attributed to.
    async fn close_gate_emitter(
        shared: &GrpcShared,
        thread: &ThreadId,
        turn: u32,
    ) -> Result<agent_server::journal::task::AgentTaskId> {
        let task = agent_server::journal::task::AgentTask::new_root_turn(
            thread.clone(),
            OffsetDateTime::UNIX_EPOCH,
            3,
        );
        let id = task.id.clone();
        shared.stores.task_store.submit_root_turn(task).await?;
        shared
            .stores
            .checkpoint_store
            .commit_checkpoint(close_gate_checkpoint(thread, id.as_str(), turn))
            .await?;
        Ok(id)
    }

    /// The same stores, but every task-row read fails.
    fn with_failing_task_reads(shared: &GrpcShared) -> Arc<GrpcShared> {
        let mut shared = shared.clone();
        shared.stores.task_store = Arc::new(CountingTaskStore {
            inner: Arc::clone(&shared.stores.task_store),
            complete_calls: Arc::new(AtomicUsize::new(0)),
            complete_with_result_calls: Arc::new(AtomicUsize::new(0)),
            submit_root_turn_idempotent_calls: Arc::new(AtomicUsize::new(0)),
            fail_get: true,
        });
        Arc::new(shared)
    }

    /// The residual the monotonic rule alone cannot see: a cancelled
    /// predecessor's late full-turn salvage commits `Done` AT the
    /// checkpoint head, so by turn number it is indistinguishable from
    /// the thread's genuine tail and closes the follower before the
    /// promoted successor commits anything. Emitter attribution
    /// suppresses that close; the identical frame WITHOUT an emitter id
    /// keeps closing, which pins that the gate is purely additive.
    #[tokio::test]
    async fn cancelled_emitters_salvage_done_never_closes() -> Result<()> {
        let (shared, thread, id_less) = close_gate_fixture("t-cancelled-emitter")?;
        shared
            .stores
            .thread_store
            .get_or_create(&thread, OffsetDateTime::UNIX_EPOCH)
            .await?;
        let emitter = close_gate_emitter(&shared, &thread, 1).await?;
        let salvage = close_gate_done_from(&thread, 9, 1, &emitter);

        // Before the cancel the very same frame is the thread's genuine
        // tail — suppression is caused by the emitter's cancellation,
        // not by carrying an id.
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &salvage).await,
            Some(pb::StreamCloseReason::ThreadCompleted),
            "a live emitter's Done at the head closes",
        );

        let outcome = shared
            .stores
            .task_store
            .cancel_tree(&emitter, OffsetDateTime::UNIX_EPOCH)
            .await?;
        assert!(
            !outcome.transitioned.is_empty(),
            "the emitter must actually cancel",
        );
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &salvage).await,
            None,
            "a cancelled emitter's late Done is salvage of superseded work; it must not close",
        );

        assert_eq!(
            confirmed_close_reason(&shared, &thread, &id_less).await,
            Some(pb::StreamCloseReason::ThreadCompleted),
            "id-less frames keep the historical behavior",
        );
        Ok(())
    }

    /// Suppression is scoped to the followed thread. A fork re-commits
    /// the source's frames verbatim, so the copy names a task that never
    /// ran on the destination; cancelling that source task supersedes
    /// work on the SOURCE thread only. Honouring it here would idle a
    /// follower of an independent thread that has no marker and no
    /// successor to close it.
    #[tokio::test]
    async fn cancelled_emitter_on_a_foreign_thread_still_closes() -> Result<()> {
        let shared = event_test_shared()?;
        let now = OffsetDateTime::UNIX_EPOCH;

        let source = ThreadId::from_string("t-foreign-source");
        shared
            .stores
            .thread_store
            .get_or_create(&source, now)
            .await?;
        let source_root = close_gate_emitter(&shared, &source, 1).await?;

        // The destination has its own head at the same turn — only the
        // emitter's thread binding distinguishes the two.
        let forked = ThreadId::from_string("t-foreign-fork");
        shared
            .stores
            .thread_store
            .get_or_create(&forked, now)
            .await?;
        shared
            .stores
            .checkpoint_store
            .commit_checkpoint(close_gate_checkpoint(&forked, source_root.as_str(), 1))
            .await?;

        let outcome = shared
            .stores
            .task_store
            .cancel_tree(&source_root, now)
            .await?;
        assert!(!outcome.transitioned.is_empty(), "the source root cancels");

        // Same frame, same cancelled emitter, two threads: suppressed on
        // the thread the emitter ran on, closing on the one it did not.
        assert_eq!(
            confirmed_close_reason(
                &shared,
                &source,
                &close_gate_done_from(&source, 9, 1, &source_root)
            )
            .await,
            None,
        );
        assert_eq!(
            confirmed_close_reason(
                &shared,
                &forked,
                &close_gate_done_from(&forked, 9, 1, &source_root)
            )
            .await,
            Some(pb::StreamCloseReason::ThreadCompleted),
            "a cancel on the source thread must not suppress the fork's close",
        );
        Ok(())
    }

    /// The same leak through the REAL fork path: fork a thread, then
    /// cancel the source root, then follow the fork. `ForkThread` copied
    /// the source's `Done` (emitter id and all), and the fork has no
    /// marker and no successor — it must close `ThreadCompleted` on that
    /// copied frame rather than idle forever.
    #[tokio::test]
    async fn forked_thread_closes_even_after_its_source_root_is_cancelled() -> Result<()> {
        let shared = event_test_shared()?;
        let control = GrpcControlService {
            shared: Arc::clone(&shared),
        };
        let now = OffsetDateTime::UNIX_EPOCH;
        let source = ThreadId::from_string("t-fork-source");

        // A source thread with one committed turn: checkpoint, aggregate,
        // and the turn's lifecycle events attributed to its root.
        shared
            .stores
            .thread_store
            .get_or_create(&source, now)
            .await?;
        let source_root = close_gate_emitter(&shared, &source, 1).await?;
        shared
            .stores
            .thread_store
            .commit_turn(
                &source,
                1,
                &agent_sdk_foundation::TokenUsage::default(),
                now,
            )
            .await?;
        commit_and_notify(
            &shared,
            &source,
            AgentEvent::start(source.clone(), 1).with_emitter_task_id(source_root.as_str()),
        )
        .await?;
        commit_and_notify(
            &shared,
            &source,
            AgentEvent::done(
                source.clone(),
                1,
                agent_sdk_foundation::TokenUsage::default(),
                std::time::Duration::from_secs(1),
            )
            .with_emitter_task_id(source_root.as_str()),
        )
        .await?;

        let forked = control
            .fork_thread(Request::new(pb::ForkThreadRequest {
                request_id: "fork-emitter-leak".into(),
                source_thread_id: source.to_string(),
                fork_after_committed_turns: 1,
                thread_id: String::new(),
            }))
            .await?
            .into_inner()
            .thread
            .and_then(|view| view.thread)
            .context("fork_thread missing snapshot")?
            .thread_id;
        let forked = ThreadId::from_string(forked);

        // The copy kept the SOURCE root's emitter id...
        let copied = journaled_terminal_frame(&shared, &forked).await?;
        assert_eq!(
            copied.event.emitter_task_id(),
            Some(source_root.as_str()),
            "the fork inherits the source's provenance verbatim",
        );

        // ...and the source root is cancelled AFTER the fork snapshotted
        // it — the interleaving that leaks across threads.
        let outcome = shared
            .stores
            .task_store
            .cancel_tree(&source_root, now)
            .await?;
        assert!(!outcome.transitioned.is_empty(), "the source root cancels");

        let mut stream = open_inline_stream(&shared, &forked, None).await?;
        let closed = loop {
            let frame = next_inline_frame(&mut stream).await?;
            if let Some(StreamItem::Closed(closed)) = frame.item {
                break closed;
            }
        };
        assert_eq!(
            closed.reason,
            pb::StreamCloseReason::ThreadCompleted as i32,
            "the fork's follower must close on its own copied Done, not idle: {closed:?}",
        );
        Ok(())
    }

    /// Fallback arms: everything except a row that reads back
    /// `Cancelled` leaves the historical disposition intact — a missing
    /// row (retention-trimmed journal, foreign id) and a store read
    /// failure both keep the close rather than strand the client.
    #[tokio::test]
    async fn emitter_reads_that_are_not_cancelled_keep_the_close() -> Result<()> {
        let (shared, thread, _) = close_gate_fixture("t-emitter-fallbacks")?;
        shared
            .stores
            .thread_store
            .get_or_create(&thread, OffsetDateTime::UNIX_EPOCH)
            .await?;
        let emitter = close_gate_emitter(&shared, &thread, 1).await?;

        let missing = agent_server::journal::task::AgentTaskId::from_string("task_never_inserted");
        assert_eq!(
            confirmed_close_reason(
                &shared,
                &thread,
                &close_gate_done_from(&thread, 9, 1, &missing)
            )
            .await,
            Some(pb::StreamCloseReason::ThreadCompleted),
            "a missing emitter row falls back to the monotonic rule",
        );

        let frame = close_gate_done_from(&thread, 9, 1, &emitter);
        let failing = with_failing_task_reads(&shared);
        assert_eq!(
            confirmed_close_reason(&failing, &thread, &frame).await,
            Some(pb::StreamCloseReason::ThreadCompleted),
            "a task-store read failure must never strand a follower",
        );
        Ok(())
    }

    /// A completed emitter is not a cancelled one: its terminal frame is
    /// the thread's answer and closes.
    #[tokio::test]
    async fn completed_emitters_done_at_head_closes() -> Result<()> {
        let (shared, thread, _) = close_gate_fixture("t-completed-emitter")?;
        shared
            .stores
            .thread_store
            .get_or_create(&thread, OffsetDateTime::UNIX_EPOCH)
            .await?;
        let emitter = close_gate_emitter(&shared, &thread, 1).await?;

        let worker = agent_server::WorkerId::from_string("w-emitter");
        let lease = agent_server::LeaseId::from_string("l-emitter");
        shared
            .stores
            .task_store
            .try_acquire_task(
                &emitter,
                worker.clone(),
                lease.clone(),
                OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(600),
                OffsetDateTime::UNIX_EPOCH,
            )
            .await?
            .context("acquire the emitter")?;
        shared
            .stores
            .task_store
            .complete_task(&emitter, &worker, &lease, OffsetDateTime::UNIX_EPOCH)
            .await?;

        assert_eq!(
            confirmed_close_reason(
                &shared,
                &thread,
                &close_gate_done_from(&thread, 9, 1, &emitter)
            )
            .await,
            Some(pb::StreamCloseReason::ThreadCompleted),
        );
        Ok(())
    }

    /// End to end over a real journal: predecessor cancelled, successor
    /// promoted and requeued past the collision. The follower must not
    /// close on the predecessor's salvaged `Done`; it stays attached
    /// through the requeue boundary and the successor's fresh `Start`,
    /// and closes only on the successor's own `Done` — the delivery the
    /// requeue path promises, now kept in-stream instead of only after a
    /// reconnect.
    #[tokio::test]
    async fn collision_requeue_closes_only_on_the_successors_done() -> Result<()> {
        let shared = event_test_shared()?;
        let thread = ThreadId::from_string("t-collision-follower");
        let fixture = seed_cancelled_root_with_salvage(&shared, &thread, true).await?;
        let successor_id = fixture.successor.context("the cancel must promote one")?;

        // The predecessor's late salvage `Done` lands at the checkpoint
        // head — the exact frame that used to close the follower.
        let salvage = journaled_terminal_frame(&shared, &thread).await?;
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &salvage).await,
            None,
            "the follower must survive the predecessor's salvage",
        );

        run_successor_to_completion(&shared, &thread, &successor_id).await?;

        // Everything the successor journaled is gated as a real row: the
        // requeue boundary and the fresh `Start` are non-terminal, the
        // predecessor's salvage stays suppressed now that the head has
        // advanced (both arms agree), and only the successor's own `Done`
        // closes.
        let events = shared.stores.event_repo.get_events(&thread).await?;
        let boundary = events
            .iter()
            .find(|committed| matches!(committed.event, AgentEvent::Error { .. }))
            .context("the requeue must journal its boundary")?;
        assert_eq!(
            confirmed_close_reason(&shared, &thread, boundary).await,
            None
        );

        let successor_start = events
            .iter()
            .rfind(|committed| matches!(committed.event, AgentEvent::Start { .. }))
            .context("the retry must journal a fresh Start")?;
        assert_eq!(
            confirmed_close_reason(&shared, &thread, successor_start).await,
            None,
        );
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &salvage).await,
            None,
        );

        let answer = journaled_terminal_frame(&shared, &thread).await?;
        assert_eq!(
            confirmed_close_reason(&shared, &thread, &answer).await,
            Some(pb::StreamCloseReason::ThreadCompleted),
            "the successor's own Done closes the follower",
        );
        Ok(())
    }

    /// A thread whose active root committed its turn, was cancelled, and
    /// whose late salvage `Done` then landed at the checkpoint head.
    struct CancelledRootFixture {
        successor: Option<agent_server::journal::task::AgentTaskId>,
        /// The cancel marker's sequence — the cursor a follower
        /// reconnects with after the marker released it.
        marker_sequence: u64,
    }

    /// Seed that thread. The marker is committed by the real cancel
    /// transaction (not hand-written), and with `queue_successor` a
    /// queued root is promoted by that same transaction.
    async fn seed_cancelled_root_with_salvage(
        shared: &GrpcShared,
        thread: &ThreadId,
        queue_successor: bool,
    ) -> Result<CancelledRootFixture> {
        let now = OffsetDateTime::UNIX_EPOCH;
        shared
            .stores
            .thread_store
            .get_or_create(thread, now)
            .await?;
        let predecessor = close_gate_emitter(shared, thread, 1).await?;
        commit_and_notify(
            shared,
            thread,
            AgentEvent::start(thread.clone(), 1).with_emitter_task_id(predecessor.as_str()),
        )
        .await?;

        let successor = if queue_successor {
            let queued =
                agent_server::journal::task::AgentTask::new_root_turn(thread.clone(), now, 3);
            let id = queued.id.clone();
            shared.stores.task_store.submit_root_turn(queued).await?;
            Some(id)
        } else {
            None
        };

        let outcome = shared
            .stores
            .task_store
            .cancel_tree(&predecessor, now)
            .await?;
        ensure!(
            !outcome.transitioned.is_empty(),
            "the root must actually cancel",
        );
        let marker_sequence = shared
            .stores
            .event_repo
            .get_events(thread)
            .await?
            .iter()
            .rev()
            .find(|committed| matches!(committed.event, AgentEvent::Cancelled { .. }))
            .context("the cancel transaction must journal a marker")?
            .sequence;

        // The cancelled root's worker commits its full turn anyway: a
        // terminal frame AT the head, attributed to a task that is now
        // Cancelled.
        commit_and_notify(
            shared,
            thread,
            AgentEvent::done(
                thread.clone(),
                1,
                agent_sdk_foundation::TokenUsage::default(),
                std::time::Duration::from_secs(1),
            )
            .with_emitter_task_id(predecessor.as_str()),
        )
        .await?;

        Ok(CancelledRootFixture {
            successor,
            marker_sequence,
        })
    }

    /// Drive the promoted successor through the collision it loses: the
    /// requeue journals its boundary atomically with the ownership CAS,
    /// then the retry commits a fresh `Start`, the turn-2 checkpoint, and
    /// its own `Done`.
    async fn run_successor_to_completion(
        shared: &GrpcShared,
        thread: &ThreadId,
        successor: &agent_server::journal::task::AgentTaskId,
    ) -> Result<()> {
        let now = OffsetDateTime::UNIX_EPOCH;
        let worker = agent_server::WorkerId::from_string("w-successor");
        let lease = agent_server::LeaseId::from_string("l-successor");
        shared
            .stores
            .task_store
            .try_acquire_task(
                successor,
                worker.clone(),
                lease.clone(),
                now + time::Duration::seconds(600),
                now,
            )
            .await?
            .context("acquire the promoted successor")?;
        let boundary = AgentEvent::error("turn-slot collision: retrying", true)
            .with_emitter_task_id(successor.as_str());
        shared
            .stores
            .task_store
            .requeue_owned_task(successor, &worker, &lease, Some(boundary), now)
            .await?;

        commit_and_notify(
            shared,
            thread,
            AgentEvent::start(thread.clone(), 2).with_emitter_task_id(successor.as_str()),
        )
        .await?;
        shared
            .stores
            .checkpoint_store
            .commit_checkpoint(close_gate_checkpoint(thread, successor.as_str(), 2))
            .await?;
        commit_and_notify(
            shared,
            thread,
            AgentEvent::done(
                thread.clone(),
                2,
                agent_sdk_foundation::TokenUsage::default(),
                std::time::Duration::from_secs(1),
            )
            .with_emitter_task_id(successor.as_str()),
        )
        .await?;
        Ok(())
    }

    /// Commit an event and publish it to live followers, as the worker
    /// does. The requeue boundary is deliberately NOT published this way:
    /// the store journals it inside the ownership CAS, so live followers
    /// reach it through the stream's gap backfill — the same path the
    /// cross-host relay takes.
    async fn commit_and_notify(
        shared: &GrpcShared,
        thread: &ThreadId,
        event: AgentEvent,
    ) -> Result<agent_server::CommittedEvent> {
        let committed = shared
            .stores
            .event_repo
            .commit_event(thread, event, OffsetDateTime::UNIX_EPOCH)
            .await?;
        shared
            .stores
            .event_notifier
            .notify(std::slice::from_ref(&committed));
        Ok(committed)
    }

    /// The thread's last journaled `Done` — a real row, sequence and all.
    async fn journaled_terminal_frame(
        shared: &GrpcShared,
        thread: &ThreadId,
    ) -> Result<agent_server::CommittedEvent> {
        shared
            .stores
            .event_repo
            .get_events(thread)
            .await?
            .iter()
            .rev()
            .find(|committed| matches!(committed.event, AgentEvent::Done { .. }))
            .cloned()
            .context("expected a journaled Done")
    }

    async fn open_inline_stream(
        shared: &Arc<GrpcShared>,
        thread: &ThreadId,
        after_sequence: Option<u64>,
    ) -> Result<EventStream> {
        let events = GrpcEventService {
            shared: Arc::clone(shared),
        };
        Ok(events
            .stream_thread_events(Request::new(pb::StreamThreadEventsRequest {
                thread_id: thread.to_string(),
                after_sequence,
                follow_mode: pb::FollowMode::ReplayAndFollow as i32,
            }))
            .await?
            .into_inner())
    }

    fn frame_payload(frame: &pb::StreamThreadEventsResponse) -> Option<&EventPayload> {
        match frame.item.as_ref() {
            Some(StreamItem::Event(envelope)) => envelope.event.as_ref(),
            _ => None,
        }
    }

    /// The end-to-end promise, driven through the REAL follower stream:
    /// a follower that reconnected past the cancel marker replays the
    /// predecessor's salvaged `Done` WITHOUT closing, stays attached
    /// through the requeue boundary and the successor's fresh `Start`,
    /// and closes `ThreadCompleted` only on the successor's own `Done`.
    /// On main this stream ended at the salvage, and the successor's
    /// answer was reachable only by reconnecting.
    #[tokio::test]
    async fn follower_stream_past_the_marker_closes_on_the_successors_done() -> Result<()> {
        let shared = event_test_shared()?;
        let thread = ThreadId::from_string("t-stream-salvage");
        let fixture = seed_cancelled_root_with_salvage(&shared, &thread, true).await?;
        let successor_id = fixture.successor.clone().context("promoted successor")?;

        let mut stream =
            open_inline_stream(&shared, &thread, Some(fixture.marker_sequence)).await?;

        let opened = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(opened.item, Some(StreamItem::ReplayOpened(_))),
            "expected replay_opened, got {opened:?}",
        );

        // The replay tail IS the suppressed frame.
        let salvage = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(frame_payload(&salvage), Some(EventPayload::Done(_))),
            "expected the salvaged Done, got {salvage:?}",
        );

        let catchup = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(catchup.item, Some(StreamItem::ReplayCatchupComplete(_))),
            "expected replay_catchup_complete, got {catchup:?}",
        );

        // The post-replay gate runs when this frame is polled, while the
        // successor has still committed NOTHING — the exact window the
        // monotonic rule cannot see through, since the salvage sits AT the
        // head. On main a `Closed(ThreadCompleted)` arrives here; the
        // stream must instead stay open with no frame at all.
        ensure!(
            tokio::time::timeout(Duration::from_millis(250), stream.next())
                .await
                .is_err(),
            "a cancelled emitter's salvage must not close the stream",
        );

        run_successor_to_completion(&shared, &thread, &successor_id).await?;

        // The boundary reaches the follower through the gap backfill.
        let boundary = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(frame_payload(&boundary), Some(EventPayload::Error(_))),
            "expected the requeue boundary, got {boundary:?}",
        );
        let start = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(frame_payload(&start), Some(EventPayload::Start(_))),
            "expected the successor's fresh Start, got {start:?}",
        );
        let answer = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(frame_payload(&answer), Some(EventPayload::Done(_))),
            "expected the successor's Done, got {answer:?}",
        );

        let closed = next_inline_frame(&mut stream).await?;
        let Some(StreamItem::Closed(closed)) = closed.item else {
            bail!("expected a closed frame, got {closed:?}");
        };
        assert_eq!(
            closed.reason,
            pb::StreamCloseReason::ThreadCompleted as i32,
            "{closed:?}",
        );
        Ok(())
    }

    /// The same journal replayed from 0 closes at its tail: the cancel
    /// marker mid-replay is not the last event, so the successor's `Done`
    /// is what the post-replay gate sees.
    #[tokio::test]
    async fn replay_from_zero_closes_on_the_successors_done() -> Result<()> {
        let shared = event_test_shared()?;
        let thread = ThreadId::from_string("t-stream-replay-tail");
        let fixture = seed_cancelled_root_with_salvage(&shared, &thread, true).await?;
        let successor_id = fixture.successor.clone().context("promoted successor")?;
        run_successor_to_completion(&shared, &thread, &successor_id).await?;

        let mut stream = open_inline_stream(&shared, &thread, None).await?;
        let mut payloads = Vec::new();
        let closed = loop {
            let frame = next_inline_frame(&mut stream).await?;
            if let Some(payload) = frame_payload(&frame) {
                payloads.push(std::mem::discriminant(payload));
            }
            if let Some(StreamItem::Closed(closed)) = frame.item {
                break closed;
            }
        };
        assert_eq!(
            closed.reason,
            pb::StreamCloseReason::ThreadCompleted as i32,
            "{closed:?}",
        );
        // The whole history was delivered before the close — the
        // predecessor's salvage did not truncate it.
        let expected = [
            EventPayload::Start(pb::StartEvent::default()),
            EventPayload::Cancelled(pb::CancelledEvent::default()),
            EventPayload::Done(pb::DoneEvent::default()),
            EventPayload::Error(pb::ErrorEvent::default()),
            EventPayload::Start(pb::StartEvent::default()),
            EventPayload::Done(pb::DoneEvent::default()),
        ]
        .iter()
        .map(std::mem::discriminant)
        .collect::<Vec<_>>();
        assert_eq!(payloads, expected, "unexpected replayed history");
        Ok(())
    }

    /// A cancel with NOTHING queued behind it: no successor ever commits,
    /// so the suppressed salvage is the journal's tail and the follower
    /// idles instead of closing. This is a deliberate behavior change —
    /// main answered `ThreadCompleted` here, which is a lie about how the
    /// thread ended; the cancel marker (already delivered to anyone
    /// following from before it) is the authoritative close, and a cursor
    /// past the marker sees trailing salvage with no further lifecycle
    /// close, exactly as `CancelledEvent` documents.
    #[tokio::test]
    async fn cancelled_root_without_successor_leaves_the_follower_open() -> Result<()> {
        let shared = event_test_shared()?;
        let thread = ThreadId::from_string("t-stream-no-successor");
        let fixture = seed_cancelled_root_with_salvage(&shared, &thread, false).await?;
        assert!(fixture.successor.is_none(), "nothing was queued");

        // Reconnected past the marker: the salvage is the replay tail.
        let mut stream =
            open_inline_stream(&shared, &thread, Some(fixture.marker_sequence)).await?;
        let opened = next_inline_frame(&mut stream).await?;
        ensure!(matches!(opened.item, Some(StreamItem::ReplayOpened(_))));
        let salvage = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(frame_payload(&salvage), Some(EventPayload::Done(_))),
            "expected the salvaged Done, got {salvage:?}",
        );
        let catchup = next_inline_frame(&mut stream).await?;
        ensure!(
            matches!(catchup.item, Some(StreamItem::ReplayCatchupComplete(_))),
            "expected replay_catchup_complete, got {catchup:?}",
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(250), stream.next())
                .await
                .is_err(),
            "the follower must idle open, not receive a ThreadCompleted close",
        );

        // Replaying from 0 reaches the same tail, with the same outcome.
        let mut from_zero = open_inline_stream(&shared, &thread, None).await?;
        loop {
            let frame = next_inline_frame(&mut from_zero).await?;
            if matches!(frame.item, Some(StreamItem::ReplayCatchupComplete(_))) {
                break;
            }
            ensure!(
                !matches!(frame.item, Some(StreamItem::Closed(_))),
                "the replay must not close on the suppressed tail, got {frame:?}",
            );
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(250), from_zero.next())
                .await
                .is_err(),
            "a replay whose tail is the suppressed salvage must stay open too",
        );
        Ok(())
    }

    /// Every lifecycle frame carries its emitter across the wire, and an
    /// unstamped frame leaves the field unset — including the
    /// `BudgetExceeded` → `DoneEvent` passthrough, which shares
    /// `map_done_event` with `Done`.
    #[test]
    fn lifecycle_frames_project_emitter_task_id() -> Result<()> {
        let thread = ThreadId::from_string("t-emitter-wire");
        let usage = TokenUsage::default();
        let unstamped = || {
            vec![
                AgentEvent::start(thread.clone(), 1),
                AgentEvent::turn_complete(1, usage.clone()),
                AgentEvent::done(thread.clone(), 1, usage.clone(), Duration::from_secs(1)),
                budget_exceeded_event(),
                AgentEvent::error("boom", true),
                AgentEvent::cancelled(1, usage.clone()),
            ]
        };

        for event in unstamped() {
            let label = format!("{event:?}");
            assert_eq!(
                wire_emitter_task_id(&event)?,
                None,
                "{label}: an unstamped frame must not invent an id",
            );
        }
        for event in unstamped() {
            let label = format!("{event:?}");
            let stamped = event.with_emitter_task_id("task_emitter");
            assert_eq!(
                wire_emitter_task_id(&stamped)?.as_deref(),
                Some("task_emitter"),
                "{label}: the emitter must survive projection",
            );
        }
        Ok(())
    }

    fn wire_emitter_task_id(event: &AgentEvent) -> Result<Option<String>> {
        Ok(match map_event_payload(event)? {
            EventPayload::Start(frame) => frame.emitter_task_id,
            EventPayload::TurnComplete(frame) => frame.emitter_task_id,
            EventPayload::Done(frame) => frame.emitter_task_id,
            EventPayload::Error(frame) => frame.emitter_task_id,
            EventPayload::Cancelled(frame) => frame.emitter_task_id,
            other => bail!("not a lifecycle frame: {other:?}"),
        })
    }

    #[test]
    fn budget_exceeded_is_terminal_for_follow_streams() {
        // `terminal_close_reason` drives stream closure, so a
        // budget-terminated thread closes with ThreadCompleted instead of
        // hanging waiting for `Done`.
        let committed = agent_server::CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: ThreadId::from_string("t-budget"),
            sequence: 7,
            timestamp: OffsetDateTime::UNIX_EPOCH,
            event: budget_exceeded_event(),
        };
        assert_eq!(
            terminal_close_reason(&committed),
            Some(pb::StreamCloseReason::ThreadCompleted)
        );
    }

    /// A fork below a terminal marker must not inherit it: `BudgetExceeded`
    /// and `Cancelled` are turn boundaries exactly like `Done`, or the fork
    /// would immediately read as completed/cancelled even though its state
    /// predates the stop.
    #[test]
    fn fork_boundary_excludes_budget_and_cancel_markers() {
        let committed = |sequence: u64, event: AgentEvent| agent_server::CommittedEvent {
            event_id: uuid::Uuid::now_v7(),
            thread_id: ThreadId::from_string("t-fork"),
            sequence,
            timestamp: OffsetDateTime::UNIX_EPOCH,
            event,
        };
        let done_turn_1 = AgentEvent::done(
            ThreadId::from_string("t-fork"),
            1,
            TokenUsage::default(),
            std::time::Duration::ZERO,
        );

        // Budget marker past the cutoff: excluded (its total_turns > fork_after).
        let mut budget_marker = budget_exceeded_event();
        if let AgentEvent::BudgetExceeded { total_turns, .. } = &mut budget_marker {
            *total_turns = 2;
        }
        let kept = events_up_to_fork_boundary(
            vec![
                committed(1, done_turn_1.clone()),
                committed(2, budget_marker),
            ],
            1,
        );
        assert_eq!(kept.len(), 1, "the budget marker must not cross the cutoff");
        assert!(matches!(kept[0], AgentEvent::Done { .. }));

        // Cancel marker past the cutoff: excluded the same way.
        let kept = events_up_to_fork_boundary(
            vec![
                committed(1, done_turn_1),
                committed(2, AgentEvent::cancelled(2, TokenUsage::default())),
            ],
            1,
        );
        assert_eq!(kept.len(), 1, "the cancel marker must not cross the cutoff");
        assert!(matches!(kept[0], AgentEvent::Done { .. }));
    }
}
