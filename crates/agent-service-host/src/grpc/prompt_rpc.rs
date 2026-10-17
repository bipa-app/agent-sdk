//! `DecideConfirmation{prompt_id}` and `ListPendingPrompts`: the control
//! plane's side of [`crate::prompts`].
use agent_sdk_foundation::{ThreadId, ToolResult};
use agent_server::journal::prompt::{
    PendingPrompt, PromptAnswer, PromptDecision, PromptOwner, PromptRecord, PromptState,
    StoredPrompt, get_prompt, pending_prompts,
};
use agent_server::journal::{AgentTask, AgentTaskId};
use agent_server::worker::ConfirmationDecision;
use time::OffsetDateTime;
use tonic::{Response, Status};
use tracing::warn;

use super::{
    DecideConfirmationResult, GrpcControlService, GrpcShared, RpcResult,
    decide_confirmation_fingerprint, decode_idempotency_result, idempotency_conflict_status,
    internal_status, map_confirmation_decision, map_json_value, map_pending_question,
    map_timestamp, pb,
};
use crate::confirmation_guard::PromptDecisionRequest;
use crate::prompts::{
    self, AnswerError, Answered, SharedPromptHandler, default_task_prompt_result,
};
use agent_server::journal::idempotency::{IdempotencyClaim, IdempotencyKind, IdempotencyRecord};

fn prompt_decision(decision: &ConfirmationDecision) -> PromptDecision {
    match decision {
        ConfirmationDecision::Approved => PromptDecision::Approved,
        ConfirmationDecision::Rejected { reason } => PromptDecision::Rejected {
            reason: reason.clone(),
        },
        ConfirmationDecision::Timeout => PromptDecision::TimedOut,
    }
}

fn answer_status(error: AnswerError) -> Status {
    match error {
        AnswerError::NotFound { .. } => Status::not_found(error.to_string()),
        AnswerError::AlreadyAnswered { .. }
        | AnswerError::Closed { .. }
        | AnswerError::TaskOwned { .. } => Status::failed_precondition(error.to_string()),
        AnswerError::Store(error) => internal_status("answering prompt")(error),
    }
}

fn task_prompt_result(
    handler: Option<&SharedPromptHandler>,
    record: &PromptRecord,
    owner: Option<&AgentTask>,
) -> ToolResult {
    handler.map_or_else(
        || default_task_prompt_result(record, owner),
        |handler| handler.task_prompt_result(record, owner),
    )
}

impl GrpcShared {
    /// Close the prompt rows owned by `task` once it stops awaiting its
    /// confirmation. A failure is logged: the startup sweep, and the host's
    /// own, close what this misses.
    pub(super) async fn close_owned_prompts(&self, task: &AgentTaskId) {
        let handler = self.prompt_handler.clone();
        let result = move |record: &PromptRecord, owner: Option<&AgentTask>| {
            task_prompt_result(handler.as_ref(), record, owner)
        };
        if let Err(error) = prompts::close_task_prompts(
            self.stores.task_store.as_ref(),
            self.stores.document_store.as_ref(),
            task,
            &result,
            OffsetDateTime::now_utc(),
        )
        .await
        {
            warn!(task.id = %task, error = %format!("{error:#}"), "closing the task's prompt rows failed");
        }
    }

    /// Withdraw stale task-owned prompts, then hand every answered host
    /// prompt to the handler again. Runs once when the transport starts.
    pub(super) async fn recover_prompts(&self) {
        let handler = self.prompt_handler.clone();
        let result = {
            let handler = handler.clone();
            move |record: &PromptRecord, owner: Option<&AgentTask>| {
                task_prompt_result(handler.as_ref(), record, owner)
            }
        };
        let now = OffsetDateTime::now_utc();
        let tasks = self.stores.task_store.as_ref();
        let documents = self.stores.document_store.as_ref();
        if let Err(error) =
            prompts::withdraw_stale_task_prompts(tasks, documents, &result, now).await
        {
            warn!(error = %format!("{error:#}"), "withdrawing stale prompt rows at start failed");
        }
        let Some(handler) = handler else {
            return;
        };
        if let Err(error) =
            prompts::redrive_answered_prompts(tasks, documents, handler.as_ref(), now).await
        {
            warn!(error = %format!("{error:#}"), "re-driving answered prompts at start failed");
        }
    }

    fn dispatch_answered(&self, prompt: agent_server::journal::StoredPrompt) {
        let Some(handler) = self.prompt_handler.clone() else {
            warn!(
                prompt.id = %prompt.record.id,
                "a host prompt was answered but no prompt handler is registered"
            );
            return;
        };
        let store = std::sync::Arc::clone(&self.stores.task_store);
        tokio::spawn(async move {
            if let Err(error) = prompts::resolve_answered(
                store.as_ref(),
                &prompt,
                handler.as_ref(),
                OffsetDateTime::now_utc(),
            )
            .await
            {
                warn!(
                    prompt.id = %prompt.record.id,
                    error = %format!("{error:#}"),
                    "answered prompt could not be applied; the next start retries it"
                );
            }
        });
    }

    async fn pending_prompt_pb(&self, prompt: &PendingPrompt) -> RpcResult<pb::PendingPrompt> {
        let kind = match prompt {
            PendingPrompt::ToolConfirmation { task_id, .. } => {
                let task = self
                    .stores
                    .task_store
                    .get(task_id)
                    .await
                    .map_err(internal_status("loading parked task"))?
                    .ok_or_else(|| Status::not_found(format!("task {task_id} is gone")))?;
                let snapshot = self.task_snapshot(&task).await?;
                let Some(pb::task_snapshot::StateDetail::AwaitingConfirmation(state)) =
                    snapshot.state_detail
                else {
                    return Err(
                        Status::internal(format!("task {task_id} is no longer parked")).into(),
                    );
                };
                pb::pending_prompt::Prompt::ToolConfirmation(state.confirmation.unwrap_or_default())
            }
            PendingPrompt::Question {
                task_id,
                thread_id,
                questions,
            } => pb::pending_prompt::Prompt::Question(pb::PendingQuestionBatch {
                task_id: task_id.to_string(),
                thread_id: thread_id.0.clone(),
                questions: questions.iter().map(map_pending_question).collect(),
            }),
            PendingPrompt::Row(row) => {
                let record = &row.record;
                let (host_name, owner_task_id, owner_thread_id) = match &record.owner {
                    PromptOwner::Host { name } => (
                        name.clone(),
                        record
                            .owner_task
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_default(),
                        String::new(),
                    ),
                    PromptOwner::Task { task_id, thread_id } => {
                        (String::new(), task_id.to_string(), thread_id.0.clone())
                    }
                };
                pb::pending_prompt::Prompt::Row(pb::PendingPromptRow {
                    id: record.id.clone(),
                    surface_thread_id: record.surface_thread.0.clone(),
                    host_name,
                    owner_task_id,
                    owner_thread_id,
                    tool_name: record.frame.name.clone(),
                    display_name: record.frame.display_name.clone(),
                    input: Some(map_json_value(&record.frame.input)?),
                    description: record.frame.description.clone(),
                    must_reach_human: record.must_reach_human,
                    opened_at: Some(map_timestamp(record.opened_at)?),
                })
            }
        };
        Ok(pb::PendingPrompt { prompt: Some(kind) })
    }
}

impl GrpcControlService {
    pub(super) async fn list_pending_prompts_rpc(
        &self,
        thread_id: &ThreadId,
    ) -> RpcResult<pb::ListPendingPromptsResponse> {
        self.shared.require_thread(thread_id).await?;
        let pending = pending_prompts(
            self.shared.stores.task_store.as_ref(),
            self.shared.stores.document_store.as_ref(),
            thread_id,
        )
        .await
        .map_err(internal_status("listing pending prompts"))?;
        let mut prompts = Vec::with_capacity(pending.len());
        for prompt in &pending {
            prompts.push(self.shared.pending_prompt_pb(prompt).await?);
        }
        Ok(pb::ListPendingPromptsResponse { prompts })
    }

    /// `DecideConfirmation` with a `prompt_id`.
    pub(super) async fn decide_prompt(
        &self,
        request: pb::DecideConfirmationRequest,
        thread_id: ThreadId,
    ) -> RpcResult<Response<pb::DecideConfirmationResponse>> {
        if !request.task_id.is_empty() {
            return Err(
                Status::invalid_argument("set either task_id or prompt_id, not both").into(),
            );
        }
        let decision = map_confirmation_decision(
            request
                .decision
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("confirmation decision is required"))?,
        )?;
        let documents = self.shared.stores.document_store.as_ref();
        let prompt = get_prompt(documents, &thread_id, &request.prompt_id)
            .await
            .map_err(internal_status("loading prompt"))?;
        if let (Some(guard), Some(prompt)) = (&self.shared.confirmation_guard, &prompt)
            && matches!(prompt.record.state, PromptState::Open)
        {
            guard
                .check_prompt(&PromptDecisionRequest::new(
                    &thread_id,
                    &prompt.record,
                    &decision,
                    request.automatic,
                ))
                .map_err(Status::failed_precondition)?;
        }

        let fingerprint = decide_confirmation_fingerprint(&request);
        let store = self.shared.stores.task_store.as_ref();
        match store
            .claim_idempotency(
                &request.request_id,
                IdempotencyKind::DecideConfirmation,
                &fingerprint,
            )
            .await
            .map_err(internal_status("decide-prompt idempotency lookup"))?
        {
            IdempotencyClaim::Conflict => {
                return Err(
                    idempotency_conflict_status("DecideConfirmation", &request.request_id).into(),
                );
            }
            IdempotencyClaim::Replay(record) => {
                let result: DecideConfirmationResult =
                    decode_idempotency_result(&record.result_json)?;
                return Ok(Response::new(
                    self.replay_decide_confirmation(&result).await?,
                ));
            }
            IdempotencyClaim::InFlight | IdempotencyClaim::Fresh => {}
        }
        let prompt = prompt.ok_or_else(|| {
            Status::not_found(format!(
                "no prompt {} on thread {thread_id}",
                request.prompt_id
            ))
        })?;

        let result = self
            .apply_prompt_decision(&request, &thread_id, &prompt, &decision)
            .await?;

        let result_json = serde_json::to_value(&result).map_err(|error| {
            Status::internal(format!(
                "encoding decide-prompt idempotency result: {error}"
            ))
        })?;
        store
            .record_idempotency(IdempotencyRecord {
                request_id: request.request_id,
                kind: IdempotencyKind::DecideConfirmation,
                fingerprint,
                result_json,
            })
            .await
            .map_err(internal_status("recording decide-prompt idempotency"))?;
        Ok(Response::new(
            self.replay_decide_confirmation(&result).await?,
        ))
    }

    async fn apply_prompt_decision(
        &self,
        request: &pb::DecideConfirmationRequest,
        thread_id: &ThreadId,
        prompt: &StoredPrompt,
        decision: &ConfirmationDecision,
    ) -> RpcResult<DecideConfirmationResult> {
        Ok(match &prompt.record.owner {
            PromptOwner::Task {
                task_id,
                thread_id: owner_thread,
            } => {
                if !matches!(prompt.record.state, PromptState::Open) {
                    return Err(Status::failed_precondition(format!(
                        "prompt {} is closed",
                        prompt.record.id
                    ))
                    .into());
                }
                self.require_awaiting_confirmation(task_id).await?;
                let mut task_request = request.clone();
                task_request.prompt_id.clear();
                task_request.task_id = task_id.to_string();
                task_request.thread_id = owner_thread.0.clone();
                let (task, parent) = self
                    .run_confirmation_decision(&task_request, owner_thread, task_id)
                    .await?;
                self.shared.close_owned_prompts(task_id).await;
                DecideConfirmationResult {
                    task_id: task.id.to_string(),
                    parent_task_id: parent.as_ref().map(|parent| parent.id.to_string()),
                    prompt: Some(prompt.record.id.clone()),
                }
            }
            PromptOwner::Host { .. } => {
                let answer = PromptAnswer {
                    decision: prompt_decision(decision),
                    request_id: request.request_id.clone(),
                    automatic: request.automatic,
                    answered_at: OffsetDateTime::now_utc(),
                };
                let answered = prompts::answer_prompt(
                    self.shared.stores.task_store.as_ref(),
                    self.shared.stores.document_store.as_ref(),
                    thread_id,
                    &prompt.record.id,
                    answer,
                    OffsetDateTime::now_utc(),
                )
                .await
                .map_err(answer_status)?;
                if let Answered::Fresh(answered) | Answered::Replay(answered) = answered
                    && matches!(answered.record.state, PromptState::Answered { .. })
                {
                    self.shared.dispatch_answered(answered);
                }
                DecideConfirmationResult {
                    task_id: String::new(),
                    parent_task_id: None,
                    prompt: Some(prompt.record.id.clone()),
                }
            }
        })
    }
}
