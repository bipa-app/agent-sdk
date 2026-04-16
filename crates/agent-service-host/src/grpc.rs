//! gRPC transport and local-daemon runtime for the durable service host.

use std::collections::{BTreeMap, HashMap};
use std::net::{Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;

use agent_sdk_core::events::AgentEvent;
use agent_sdk_core::llm::{self, ContentBlock, ContentSource};
use agent_sdk_core::{ThreadId, TokenUsage, ToolResult, ToolTier};
use agent_server::journal::execution_intent::GuardedExecutionDeps;
use agent_server::journal::task::{
    AgentTask, AgentTaskId, LeaseId, SubmittedInputItem, TaskKind as JournalTaskKind,
    TaskStatus as JournalTaskStatus, WorkerId,
};
use agent_server::journal::task_state::TaskState;
use agent_server::worker::{
    AgentDefinitionRegistry, ConfirmationDecision, ConfirmationResumeOutcome,
    apply_confirmation_decision, resolve_tool_bootstrap, resume_confirmed_tool,
};
use anyhow::{Context, Result};
use async_stream::try_stream;
use base64::Engine as _;
use futures::Stream;
use prost::Message as ProstMessage;
use prost_types::{
    Duration as ProtoDuration, ListValue as ProtoListValue, Struct as ProtoStruct,
    Timestamp as ProtoTimestamp, Value as ProtoValue, value::Kind as ProtoValueKind,
};
use time::OffsetDateTime;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

use crate::config::ServiceConfig;
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

#[derive(Default)]
struct IdempotencyState {
    create_thread: HashMap<String, String>,
    submit_work: HashMap<String, SubmitWorkRecord>,
    decide_confirmation: HashMap<String, DecideConfirmationRecord>,
}

#[derive(Clone)]
struct SubmitWorkRecord {
    fingerprint: Vec<u8>,
    thread_id: String,
    task_id: String,
}

#[derive(Clone)]
struct DecideConfirmationRecord {
    fingerprint: Vec<u8>,
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
    idempotency: Arc<Mutex<IdempotencyState>>,
}

impl GrpcShared {
    fn new(
        stores: StoreRegistry,
        runtime: Arc<ExecutionRuntime>,
        health: Arc<HealthSurface>,
        shutdown: CancellationToken,
        lease_duration: time::Duration,
    ) -> Self {
        Self {
            stores,
            runtime,
            health,
            shutdown,
            lease_duration,
            idempotency: Arc::new(Mutex::new(IdempotencyState::default())),
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
        let events = self
            .stores
            .event_repo
            .get_events(&thread.thread_id)
            .await
            .map_err(internal_status("loading thread events"))?;

        Ok(pb::ThreadSnapshot {
            thread_id: thread.thread_id.0.clone(),
            status: map_thread_status(thread.status),
            committed_turns: thread.committed_turns,
            total_usage: Some(map_token_usage(&thread.total_usage)),
            created_at: Some(map_timestamp(thread.created_at)?),
            updated_at: Some(map_timestamp(thread.updated_at)?),
            latest_event_sequence: events.last().map(|event| event.sequence),
            earliest_available_event_sequence: events.first().map(|event| event.sequence),
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

        let bootstrap = resolve_tool_bootstrap(acquired, self.stores.task_store.as_ref())
            .await
            .context("bootstrapping approved confirmation task")?;
        let executor_bootstrap = bootstrap.clone();
        let publish_thread_id = bootstrap.thread_id.clone();
        let running_task_id = bootstrap.task_id.clone();
        let running_worker_id = bootstrap.worker_id.clone();
        let running_lease_id = bootstrap.lease_id.clone();
        let tool_executor = Arc::clone(self.runtime.tool_executor());
        let cancel = self.shutdown.clone();
        let deps = GuardedExecutionDeps {
            task_store: self.stores.task_store.as_ref(),
            intent_store: self.stores.execution_intent_store.as_ref(),
            event_repo: self.stores.event_repo.as_ref(),
        };

        match resume_confirmed_tool(
            bootstrap,
            &deps,
            self.runtime.confirmation_policy().as_ref(),
            &self.shutdown,
            move |_tool_call, collector| {
                let tool_executor = Arc::clone(&tool_executor);
                let executor_bootstrap = executor_bootstrap.clone();
                let cancel = cancel.clone();
                async move {
                    tool_executor
                        .execute_tool_call(&executor_bootstrap, collector, cancel)
                        .await
                }
            },
            now,
        )
        .await
        {
            Ok(
                ConfirmationResumeOutcome::Executed(_)
                | ConfirmationResumeOutcome::PolicyDenied { .. },
            ) => {}
            Err(error) => {
                warn!(?error, task_id = %task_id, "approved confirmation resume failed");
                let _ = self
                    .stores
                    .task_store
                    .fail_task(
                        &running_task_id,
                        &running_worker_id,
                        &running_lease_id,
                        format!("{error:#}"),
                        now,
                    )
                    .await;
            }
        }

        self.publish_committed_after(&publish_thread_id, watermark)
            .await
            .context("publishing approved confirmation events")?;
        Ok(())
    }

    async fn publish_committed_after(
        &self,
        thread_id: &ThreadId,
        start_sequence: u64,
    ) -> Result<()> {
        let events = self
            .stores
            .event_repo
            .get_events(thread_id)
            .await
            .context("loading committed events for publish")?;
        let to_publish: Vec<_> = events
            .into_iter()
            .filter(|event| event.sequence >= start_sequence)
            .collect();
        if !to_publish.is_empty() {
            self.stores.event_notifier.notify(&to_publish);
        }
        Ok(())
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
        request: &pb::DecideConfirmationRequest,
        fingerprint: &[u8],
    ) -> RpcResult<Option<pb::DecideConfirmationResponse>> {
        let record = {
            let idempotency = self.shared.idempotency.lock().await;
            idempotency
                .decide_confirmation
                .get(&request.request_id)
                .cloned()
        };
        let Some(record) = record else {
            return Ok(None);
        };
        if record.fingerprint != fingerprint {
            return Err(
                idempotency_conflict_status("DecideConfirmation", &request.request_id).into(),
            );
        }

        let task = self
            .shared
            .stores
            .task_store
            .get(&AgentTaskId::from_string(record.task_id))
            .await
            .map_err(internal_status("loading idempotent confirmation task"))?
            .ok_or_else(|| not_found_status("task", "idempotent confirmation result"))?;
        let parent_task = match record.parent_task_id {
            Some(parent_task_id) => self
                .shared
                .stores
                .task_store
                .get(&AgentTaskId::from_string(parent_task_id))
                .await
                .map_err(internal_status("loading idempotent confirmation parent"))?,
            None => None,
        };

        Ok(Some(
            self.confirmation_response(&task, parent_task.as_ref())
                .await?,
        ))
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

        let existing = {
            let idempotency = self.shared.idempotency.lock().await;
            idempotency.create_thread.get(&request.request_id).cloned()
        };
        if let Some(thread_id) = existing {
            let response = pb::CreateThreadResponse {
                thread: Some(
                    self.shared
                        .thread_view(&ThreadId::from_string(thread_id))
                        .await?,
                ),
            };
            return Ok(Response::new(response));
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

        {
            let mut idempotency = self.shared.idempotency.lock().await;
            idempotency
                .create_thread
                .insert(request.request_id, thread_id.0.clone());
        }

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
        let request = request.into_inner();
        require_request_id(&request.request_id)?;
        let thread_id = parse_thread_id(&request.thread_id)?;
        let fingerprint = submit_work_fingerprint(&request);

        if let Some(record) = {
            let idempotency = self.shared.idempotency.lock().await;
            idempotency.submit_work.get(&request.request_id).cloned()
        } {
            if record.fingerprint != fingerprint || record.thread_id != request.thread_id {
                return Err(idempotency_conflict_status(
                    "SubmitThreadWork",
                    &request.request_id,
                ));
            }

            let task = self
                .shared
                .stores
                .task_store
                .get(&AgentTaskId::from_string(record.task_id))
                .await
                .map_err(internal_status("loading idempotent submitted task"))?
                .ok_or_else(|| not_found_status("task", "idempotent submit result"))?;

            return Ok(Response::new(pb::SubmitThreadWorkResponse {
                thread: Some(self.shared.thread_view(&thread_id).await?),
                task: Some(self.shared.task_snapshot(&task).await?),
            }));
        }

        let thread = self.shared.require_thread(&thread_id).await?;
        if thread.status.is_completed() {
            return Err(Status::failed_precondition(format!(
                "thread {} is completed",
                thread.thread_id
            )));
        }

        if request.input.is_empty() {
            return Err(Status::invalid_argument(
                "submit_thread_work requires at least one input item",
            ));
        }

        let submitted_input = request
            .input
            .iter()
            .map(map_submitted_input_item)
            .collect::<Result<Vec<_>, _>>()?;

        let now = OffsetDateTime::now_utc();
        let submission_template =
            AgentTask::new_root_turn(thread_id.clone(), now, AgentTask::DEFAULT_MAX_ATTEMPTS);
        let definition = self
            .shared
            .stores
            .definition_registry
            .resolve(&submission_template)
            .await
            .map_err(internal_status(
                "resolving root-turn definition for submission",
            ))?;
        let task = AgentTask::new_root_turn_with_input(
            thread_id.clone(),
            submitted_input,
            now,
            definition.policy.max_attempts,
        );
        let task = self
            .shared
            .stores
            .task_store
            .submit_root_turn(task)
            .await
            .map_err(internal_status("submitting root turn"))?;

        {
            let mut idempotency = self.shared.idempotency.lock().await;
            idempotency.submit_work.insert(
                request.request_id,
                SubmitWorkRecord {
                    fingerprint,
                    thread_id: thread_id.0.clone(),
                    task_id: task.id.to_string(),
                },
            );
        }

        Ok(Response::new(pb::SubmitThreadWorkResponse {
            thread: Some(self.shared.thread_view(&thread_id).await?),
            task: Some(self.shared.task_snapshot(&task).await?),
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

        if let Some(response) = self
            .replay_decide_confirmation(&request, &fingerprint)
            .await?
        {
            return Ok(Response::new(response));
        }

        let (task, parent_task) = self
            .run_confirmation_decision(&request, &thread_id, &task_id)
            .await?;

        {
            let mut idempotency = self.shared.idempotency.lock().await;
            idempotency.decide_confirmation.insert(
                request.request_id,
                DecideConfirmationRecord {
                    fingerprint,
                    task_id: task.id.to_string(),
                    parent_task_id: parent_task.as_ref().map(|parent| parent.id.to_string()),
                },
            );
        }

        Ok(Response::new(
            self.confirmation_response(&task, parent_task.as_ref())
                .await?,
        ))
    }
}

type EventStream =
    Pin<Box<dyn Stream<Item = Result<pb::StreamThreadEventsResponse, Status>> + Send>>;

struct ReplayState {
    all_events: Vec<agent_server::CommittedEvent>,
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
            let mut live_rx = shared.stores.event_notifier.subscribe(&thread_id);
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

            let replay_events = select_replay_events(replay.all_events, after_sequence);
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

            let last_replayed_done = replay_events.last().is_some_and(is_done_event);
            if follow_mode == pb::FollowMode::ReplayOnly {
                yield closed_stream_response(
                    &thread_id,
                    pb::StreamCloseReason::ReplayExhausted,
                    last_delivered_sequence,
                );
                return;
            }
            if last_replayed_done {
                yield closed_stream_response(
                    &thread_id,
                    pb::StreamCloseReason::ThreadCompleted,
                    last_delivered_sequence,
                );
                return;
            }

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
                            if last_delivered_sequence.is_some_and(|sequence| event.sequence <= sequence) {
                                continue;
                            }
                            last_delivered_sequence = Some(event.sequence);
                            let is_done = is_done_event(&event);
                            yield event_stream_response(&event)?;
                            if is_done {
                                yield closed_stream_response(
                                    &thread_id,
                                    pb::StreamCloseReason::ThreadCompleted,
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
        };

        Box::pin(stream)
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
    let next_sequence = shared
        .stores
        .event_repo
        .next_sequence(thread_id)
        .await
        .map_err(as_stream_status("capturing event watermark"))?;
    let all_events = shared
        .stores
        .event_repo
        .get_events(thread_id)
        .await
        .map_err(as_stream_status("loading replay events"))?;

    Ok(ReplayState {
        earliest_available: all_events.first().map(|event| event.sequence),
        latest_available: all_events.last().map(|event| event.sequence),
        all_events,
        next_sequence,
    })
}

const fn has_retention_gap(after_sequence: Option<u64>, earliest_available: Option<u64>) -> bool {
    matches!(
        (after_sequence, earliest_available),
        (Some(requested_after), Some(earliest)) if requested_after < earliest.saturating_sub(1)
    )
}

fn select_replay_events(
    all_events: Vec<agent_server::CommittedEvent>,
    after_sequence: Option<u64>,
) -> Vec<agent_server::CommittedEvent> {
    match after_sequence {
        Some(after_sequence) => all_events
            .into_iter()
            .filter(|event| event.sequence > after_sequence)
            .collect(),
        None => all_events,
    }
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
    #[must_use]
    pub fn new(
        stores: StoreRegistry,
        runtime: Arc<ExecutionRuntime>,
        health: Arc<HealthSurface>,
        shutdown: CancellationToken,
        lease_duration: time::Duration,
    ) -> Self {
        Self {
            shared: Arc::new(GrpcShared::new(
                stores,
                runtime,
                health,
                shutdown,
                lease_duration,
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

        Server::builder()
            .add_service(health_service)
            .add_service(AgentControlServiceServer::new(control_service))
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
        let stores = StoreRegistry::from_config(&config.storage, definition_registry)
            .context("creating daemon stores")?;
        stores
            .initialize()
            .await
            .context("initializing daemon stores")?;
        let host = ServiceHost::with_stores(config.clone(), stores.clone(), Arc::clone(&runtime))
            .context("creating daemon host")?;
        let health = Arc::clone(host.health());
        let shutdown = host.shutdown_token();
        let grpc = GrpcTransport::new(
            stores,
            runtime,
            health,
            shutdown.clone(),
            config.worker.lease_duration(),
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

fn require_request_id(request_id: &str) -> RpcResult<()> {
    if request_id.trim().is_empty() {
        return Err(Status::invalid_argument("request_id is required").into());
    }
    Ok(())
}

fn parse_thread_id(thread_id: &str) -> RpcResult<ThreadId> {
    if thread_id.trim().is_empty() {
        return Err(Status::invalid_argument("thread_id is required").into());
    }
    Ok(ThreadId::from_string(thread_id))
}

fn parse_task_id(task_id: &str) -> RpcResult<AgentTaskId> {
    if task_id.trim().is_empty() {
        return Err(Status::invalid_argument("task_id is required").into());
    }
    Ok(AgentTaskId::from_string(task_id))
}

fn internal_status(
    context: &'static str,
) -> impl Fn(anyhow::Error) -> Status + Copy + Send + Sync + 'static {
    move |error| Status::internal(format!("{context}: {error:#}"))
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

fn submit_work_fingerprint(request: &pb::SubmitThreadWorkRequest) -> Vec<u8> {
    let mut request = request.clone();
    request.request_id.clear();
    request.encode_to_vec()
}

fn decide_confirmation_fingerprint(request: &pb::DecideConfirmationRequest) -> Vec<u8> {
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
    operation: &agent_sdk_core::ListenExecutionContext,
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
    map_lifecycle_event_payload(event)
}

fn map_message_event_payload(event: &AgentEvent) -> RpcResult<Option<pb::event_envelope::Event>> {
    match event {
        AgentEvent::Start { thread_id, turn } => {
            Ok(Some(pb::event_envelope::Event::Start(pb::StartEvent {
                thread_id: thread_id.0.clone(),
                turn: map_u64(*turn, "turn")?,
            })))
        }
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

fn map_lifecycle_event_payload(event: &AgentEvent) -> RpcResult<pb::event_envelope::Event> {
    match event {
        AgentEvent::TurnComplete { turn, usage } => Ok(pb::event_envelope::Event::TurnComplete(
            pb::TurnCompleteEvent {
                turn: map_u64(*turn, "turn")?,
                usage: Some(map_token_usage(usage)),
            },
        )),
        AgentEvent::Done {
            thread_id,
            total_turns,
            total_usage,
            duration,
        } => Ok(pb::event_envelope::Event::Done(pb::DoneEvent {
            thread_id: thread_id.0.clone(),
            total_turns: map_u64(*total_turns, "total_turns")?,
            total_usage: Some(map_token_usage(total_usage)),
            duration: Some(map_duration(*duration)?),
        })),
        AgentEvent::Error {
            message,
            recoverable,
        } => Ok(pb::event_envelope::Event::Error(pb::ErrorEvent {
            message: message.clone(),
            recoverable: *recoverable,
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
        AgentEvent::SubagentProgress {
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
        } => Ok(pb::event_envelope::Event::SubagentProgress(
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
        )),
        _ => Err(Status::internal("unsupported event variant").into()),
    }
}

fn map_u64<T>(value: T, field: &'static str) -> RpcResult<u64>
where
    T: TryInto<u64>,
{
    value
        .try_into()
        .map_err(|_| Status::internal(format!("{field} out of range for protobuf")).into())
}

const fn is_done_event(event: &agent_server::CommittedEvent) -> bool {
    matches!(event.event, AgentEvent::Done { .. })
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::*;
    use crate::runtime::{
        AllowAllConfirmationPolicy, ExecutionRuntime, NoopToolExecutor, StaticProviderResolver,
        ToolCallExecutor,
    };
    #[cfg(feature = "postgres")]
    use agent_sdk_core::ThreadId;
    use agent_sdk_core::llm::{ChatOutcome, ChatRequest, ChatResponse, StopReason, Tool, Usage};
    use agent_sdk_providers::LlmProvider;
    #[cfg(feature = "postgres")]
    use agent_server::AgentTaskId;
    use agent_server::worker::definition::{AgentDefinition, RuntimePolicy, ThinkingPolicy};
    use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;
    use anyhow::{Context, Result, anyhow, bail};
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
                let _ = sqlx::query(&format!("DROP SCHEMA IF EXISTS {schema} CASCADE"))
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
        sqlx::query(&format!("CREATE SCHEMA {schema}"))
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
}
