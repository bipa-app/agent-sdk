use crate::stores::ToolExecutionStore;
use crate::types::{ExecutionStatus, PendingToolCallInfo, ThreadId, ToolExecution, ToolResult};
use log::warn;
use std::sync::Arc;
use time::OffsetDateTime;

/// Execute a tool with idempotency tracking via the execution store.
///
/// Records execution start before running the tool and completion after,
/// enabling crash recovery and deduplication.
pub(super) async fn execute_with_idempotency<Fut>(
    execution_store: Option<&Arc<dyn ToolExecutionStore>>,
    pending: &PendingToolCallInfo,
    thread_id: &ThreadId,
    execute: Fut,
) -> ToolResult
where
    Fut: std::future::Future<Output = ToolResult>,
{
    let started_at = OffsetDateTime::now_utc();
    record_execution_start(execution_store, pending, thread_id, started_at).await;
    let result = execute.await;
    record_execution_complete(execution_store, pending, thread_id, &result, started_at).await;
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
