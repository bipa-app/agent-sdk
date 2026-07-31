use crate::stores::ToolExecutionStore;
use crate::types::{
    AgentError, ExecutionStatus, PendingToolCallInfo, ThreadId, ToolExecution, ToolResult,
};
use agent_sdk_tools::artifacts::ArtifactStore;
use log::warn;
use std::sync::Arc;
use time::OffsetDateTime;

/// Execute a tool with idempotency tracking via the execution store.
///
/// Records execution start before running the tool and completion after,
/// enabling crash recovery and deduplication.
///
/// This is the SDK's single tool-result choke point: every sync, async,
/// listen, and confirmation-resume execution funnels through here, so the
/// shared inline output budget is enforced once, before the result is
/// committed to the execution store, audited, journaled as
/// `tool_call_end`, or appended to the transcript. Over-budget output is
/// spilled byte-identical to `artifact_store` and replaced inline by a
/// bounded head + tail window plus the `[raw output: artifact://<id>]`
/// recovery footer. Without a store the result passes through unchanged
/// (per-tool caps remain the only bound).
pub(super) async fn execute_with_idempotency<Fut>(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    artifact_store: Option<&Arc<ArtifactStore>>,
    pending: &PendingToolCallInfo,
    thread_id: &ThreadId,
    execute: Fut,
) -> Result<ToolResult, AgentError>
where
    Fut: Future<Output = Result<ToolResult, AgentError>>,
{
    let started_at = OffsetDateTime::now_utc();
    record_execution_start(execution_store, pending, thread_id, started_at).await;
    let mut result = execute.await;
    if let (Ok(tool_result), Some(store)) = (&mut result, artifact_store)
        && let Err(error) = store.apply_inline_budget(tool_result, &pending.name)
    {
        // A failed spill never destroys bytes: the result passes through
        // uncapped, exactly as if no store were configured.
        warn!(
            "Failed to spill over-budget tool output; returning it inline (tool_call_id={}, tool_name={}, error={error:#})",
            pending.id, pending.name
        );
    }
    if let Ok(tool_result) = &result {
        record_execution_complete(execution_store, pending, thread_id, tool_result, started_at)
            .await;
    }
    result
}

/// Check for an existing completed execution and return cached result.
///
/// Returns `Some(result)` if the execution was completed, `None` if not found
/// or still in-flight.
pub(super) async fn try_get_cached_result(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    tool_call_id: &str,
) -> Option<ToolResult> {
    let store = execution_store?;
    let execution = store.get_execution(tool_call_id).await.ok()??;

    match execution.status {
        ExecutionStatus::Completed => execution.result,
        ExecutionStatus::InFlight => {
            // Log warning that we found an in-flight execution
            // This means a previous attempt crashed mid-execution
            warn!(
                "Found in-flight execution from previous attempt, re-executing (tool_call_id={}, tool_name={})",
                tool_call_id, execution.tool_name
            );
            None
        }
    }
}

/// Record that we're about to start executing a tool (write-ahead).
pub(super) async fn record_execution_start(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    pending: &PendingToolCallInfo,
    thread_id: &ThreadId,
    started_at: OffsetDateTime,
) {
    if let Some(store) = execution_store {
        let execution = ToolExecution::new_in_flight(
            &pending.id,
            thread_id.clone(),
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            started_at,
        );
        if let Err(e) = store.record_execution(execution).await {
            warn!(
                "Failed to record execution start (tool_call_id={}, error={})",
                pending.id, e
            );
        }
    }
}

/// Record that tool execution completed.
pub(super) async fn record_execution_complete(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    pending: &PendingToolCallInfo,
    thread_id: &ThreadId,
    result: &ToolResult,
    started_at: OffsetDateTime,
) {
    if let Some(store) = execution_store {
        let mut execution = ToolExecution::new_in_flight(
            &pending.id,
            thread_id.clone(),
            &pending.name,
            &pending.display_name,
            pending.input.clone(),
            started_at,
        );
        execution.complete(result.clone());
        if let Err(e) = store.update_execution(execution).await {
            warn!(
                "Failed to record execution completion (tool_call_id={}, error={})",
                pending.id, e
            );
        }
    }
}
