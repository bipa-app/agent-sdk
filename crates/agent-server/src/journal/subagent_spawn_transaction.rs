//! Atomic durable subagent-spawn commit hook.

use agent_sdk_foundation::events::AgentEvent;
use anyhow::Result;
use async_trait::async_trait;
use time::OffsetDateTime;

use super::committed_event::CommittedEvent;
use super::store::SubagentInvocationSpawn;
use super::task::{AgentTask, AgentTaskId, LeaseId, WorkerId};
use super::thread::Thread;

/// Complete write set for one durable subagent spawn.
pub struct SubagentSpawnCommit {
    /// Boundary-injection rows this turn already delivered; completed
    /// Queued→Completed inside the same transaction as the spawn.
    pub delivered_injection_ids: Vec<AgentTaskId>,
    pub parent_id: AgentTaskId,
    pub worker: WorkerId,
    pub lease: LeaseId,
    pub spawn: SubagentInvocationSpawn,
    pub event: SubagentSpawnEvent,
    pub outbox_max_attempts: u32,
    pub now: OffsetDateTime,
}

/// Event fields known before the store mints the invocation and child-root ids.
pub struct SubagentSpawnEvent {
    pub subagent_id: String,
    pub subagent_name: String,
}

/// Rows returned from a successful atomic subagent spawn.
pub struct SubagentSpawnOutcome {
    pub parent_task: AgentTask,
    pub invocation_task: AgentTask,
    pub child_thread: Thread,
    pub child_root_task: AgentTask,
    pub committed_event: CommittedEvent,
}

/// Build the parent-visible start event from the invocation's DURABLE
/// linkage — for the batch/mixed paths, where the spawn struct has
/// already been consumed by row construction when events are built.
///
/// # Errors
/// Returns an error if the invocation row carries no
/// [`crate::journal::TaskState::SubagentInvocation`] linkage.
pub fn started_event_from_invocation(
    event: SubagentSpawnEvent,
    invocation: &AgentTask,
    child_root: &AgentTask,
) -> Result<AgentEvent> {
    let linkage = invocation
        .state
        .subagent_invocation()
        .ok_or_else(|| anyhow::anyhow!("invocation {} missing durable linkage", invocation.id))?;
    Ok(AgentEvent::SubagentProgress {
        subagent_id: event.subagent_id,
        subagent_name: event.subagent_name.clone(),
        nickname: linkage.spec.nickname.clone(),
        child_thread_id: Some(linkage.child_thread_id.clone()),
        child_root_task_id: Some(child_root.id.to_string()),
        subagent_task_id: Some(invocation.id.to_string()),
        max_turns: Some(linkage.spec.max_turns),
        current_turn: Some(0),
        model: Some(linkage.spec.model.clone()),
        tool_name: event.subagent_name,
        tool_context: linkage.spec.task.clone(),
        completed: false,
        success: false,
        tool_count: 0,
        total_tokens: 0,
        input_tokens: 0,
        output_tokens: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    })
}

/// Backend hook that creates child-thread/task rows, the parent progress event,
/// and its coalesced outbox advisory in one transaction.
#[async_trait]
pub trait AtomicSubagentSpawnCommitter: Send + Sync {
    async fn commit_subagent_spawn_atomic(
        &self,
        params: SubagentSpawnCommit,
    ) -> Result<SubagentSpawnOutcome>;
}

/// Build the parent-visible start event after the store has minted linked ids.
#[must_use]
pub fn subagent_started_event(
    event: SubagentSpawnEvent,
    spawn: &SubagentInvocationSpawn,
    invocation: &AgentTask,
    child_root: &AgentTask,
) -> AgentEvent {
    AgentEvent::SubagentProgress {
        subagent_id: event.subagent_id,
        subagent_name: event.subagent_name.clone(),
        nickname: spawn.spec.nickname.clone(),
        child_thread_id: Some(spawn.child_thread_id.clone()),
        child_root_task_id: Some(child_root.id.to_string()),
        subagent_task_id: Some(invocation.id.to_string()),
        max_turns: Some(spawn.spec.max_turns),
        current_turn: Some(0),
        model: Some(spawn.spec.model.clone()),
        tool_name: event.subagent_name,
        tool_context: spawn.spec.task.clone(),
        completed: false,
        success: false,
        tool_count: 0,
        total_tokens: 0,
        input_tokens: 0,
        output_tokens: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    }
}
