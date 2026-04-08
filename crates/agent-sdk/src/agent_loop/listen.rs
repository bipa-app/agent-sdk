use super::helpers::send_event;
use super::types::{
    LISTEN_TOTAL_TIMEOUT, LISTEN_UPDATE_TIMEOUT, ListenReady, ListenUpdateHandling,
    MAX_LISTEN_UPDATES,
};
use crate::events::{AgentEvent, SequenceCounter};
use crate::hooks::AgentHooks;
use crate::stores::EventStore;
use crate::tools::{ErasedListenTool, ListenStopReason, ListenToolUpdate, ToolContext};
use crate::types::{AgentError, PendingToolCallInfo, ThreadId, ToolResult};
use futures::StreamExt;
use log::warn;
use std::sync::Arc;
use std::time::Instant;
use time::OffsetDateTime;
use tokio::time::timeout;

pub(super) fn build_listen_progress_data(
    operation_id: &str,
    revision: u64,
    snapshot: Option<&serde_json::Value>,
    expires_at: Option<OffsetDateTime>,
) -> serde_json::Value {
    let mut data = serde_json::json!({
        "operation_id": operation_id,
        "revision": revision,
        "expires_at": expires_at,
    });

    if let Some(snapshot) = snapshot
        && let Some(object) = data.as_object_mut()
    {
        object.insert("snapshot".to_string(), snapshot.clone());
    }

    data
}

pub(super) fn build_listen_confirmation_input(
    original_input: &serde_json::Value,
    ready: &ListenReady,
) -> serde_json::Value {
    serde_json::json!({
        "requested_input": original_input,
        "prepared_snapshot": ready.snapshot,
        "operation_id": ready.operation_id,
        "revision": ready.revision,
        "expires_at": ready.expires_at,
    })
}

pub(super) enum ListenWaitError {
    Tool(ToolResult),
    Event(AgentError),
}

pub(super) async fn cancel_listen_with_warning<Ctx>(
    tool: &Arc<dyn ErasedListenTool<Ctx>>,
    tool_context: &ToolContext<Ctx>,
    operation_id: &str,
    reason: ListenStopReason,
    tool_call_id: &str,
    tool_name: &str,
) where
    Ctx: Send + Sync + Clone + 'static,
{
    if let Err(err) = tool.cancel(tool_context, operation_id, reason).await {
        warn!(
            "Failed to cancel listen operation (tool_call_id={tool_call_id}, tool_name={tool_name}, operation_id={operation_id}, reason={reason:?}, error={err})"
        );
    }
}

pub(super) async fn cancel_last_listen_operation<Ctx>(
    tool: &Arc<dyn ErasedListenTool<Ctx>>,
    tool_context: &ToolContext<Ctx>,
    pending: &PendingToolCallInfo,
    last_operation_id: Option<&str>,
    reason: ListenStopReason,
) where
    Ctx: Send + Sync + Clone + 'static,
{
    if let Some(operation_id) = last_operation_id {
        cancel_listen_with_warning(
            tool,
            tool_context,
            operation_id,
            reason,
            &pending.id,
            &pending.name,
        )
        .await;
    }
}

pub(super) async fn handle_listen_update<H>(
    pending: &PendingToolCallInfo,
    hooks: &Arc<H>,
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    turn: usize,
    seq: &SequenceCounter,
    update: ListenToolUpdate,
    last_operation_id: &mut Option<String>,
) -> Result<ListenUpdateHandling, ListenWaitError>
where
    H: AgentHooks,
{
    match update {
        ListenToolUpdate::Listening {
            operation_id,
            revision,
            message,
            snapshot,
            expires_at,
        } => {
            *last_operation_id = Some(operation_id.clone());
            let data = Some(build_listen_progress_data(
                &operation_id,
                revision,
                snapshot.as_ref(),
                expires_at,
            ));
            send_event(
                event_store,
                thread_id,
                turn,
                hooks,
                seq,
                AgentEvent::tool_progress(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    "listen_update",
                    message,
                    data,
                ),
            )
            .await
            .map_err(ListenWaitError::Event)?;
            Ok(ListenUpdateHandling::Continue)
        }
        ListenToolUpdate::Ready {
            operation_id,
            revision,
            message,
            snapshot,
            expires_at,
        } => {
            let data = Some(build_listen_progress_data(
                &operation_id,
                revision,
                Some(&snapshot),
                expires_at,
            ));
            send_event(
                event_store,
                thread_id,
                turn,
                hooks,
                seq,
                AgentEvent::tool_progress(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    "listen_ready",
                    message,
                    data,
                ),
            )
            .await
            .map_err(ListenWaitError::Event)?;
            Ok(ListenUpdateHandling::Ready(ListenReady {
                operation_id,
                revision,
                snapshot,
                expires_at,
            }))
        }
        ListenToolUpdate::Invalidated {
            operation_id,
            message,
            recoverable,
        } => {
            let data = Some(serde_json::json!({
                "operation_id": operation_id,
                "recoverable": recoverable,
            }));
            send_event(
                event_store,
                thread_id,
                turn,
                hooks,
                seq,
                AgentEvent::tool_progress(
                    &pending.id,
                    &pending.name,
                    &pending.display_name,
                    "listen_invalidated",
                    message.clone(),
                    data,
                ),
            )
            .await
            .map_err(ListenWaitError::Event)?;

            let prefix = if recoverable {
                "Listen operation invalidated (recoverable)"
            } else {
                "Listen operation invalidated"
            };
            Err(ListenWaitError::Tool(ToolResult::error(format!(
                "{prefix}: {message}"
            ))))
        }
    }
}

pub(super) async fn wait_for_listen_ready<Ctx, H>(
    pending: &PendingToolCallInfo,
    tool: &Arc<dyn ErasedListenTool<Ctx>>,
    tool_context: &ToolContext<Ctx>,
    hooks: &Arc<H>,
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    turn: usize,
    seq: &SequenceCounter,
) -> Result<ListenReady, ListenWaitError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    wait_for_listen_ready_inner(
        pending,
        tool,
        tool_context,
        hooks,
        event_store,
        thread_id,
        turn,
        seq,
    )
    .await
}

async fn wait_for_listen_ready_inner<Ctx, H>(
    pending: &PendingToolCallInfo,
    tool: &Arc<dyn ErasedListenTool<Ctx>>,
    tool_context: &ToolContext<Ctx>,
    hooks: &Arc<H>,
    event_store: &Arc<dyn EventStore>,
    thread_id: &ThreadId,
    turn: usize,
    seq: &SequenceCounter,
) -> Result<ListenReady, ListenWaitError>
where
    Ctx: Send + Sync + Clone + 'static,
    H: AgentHooks,
{
    let mut updates = tool.listen_stream(tool_context, pending.input.clone());
    let mut update_count = 0usize;
    let mut last_operation_id: Option<String> = None;
    let listen_started_at = Instant::now();

    loop {
        if listen_started_at.elapsed() >= LISTEN_TOTAL_TIMEOUT {
            cancel_last_listen_operation(
                tool,
                tool_context,
                pending,
                last_operation_id.as_deref(),
                ListenStopReason::StreamEnded,
            )
            .await;
            return Err(ListenWaitError::Tool(ToolResult::error(format!(
                "Listen tool exceeded wall-clock timeout ({}s)",
                LISTEN_TOTAL_TIMEOUT.as_secs()
            ))));
        }

        let Ok(next_update) = timeout(LISTEN_UPDATE_TIMEOUT, updates.next()).await else {
            cancel_last_listen_operation(
                tool,
                tool_context,
                pending,
                last_operation_id.as_deref(),
                ListenStopReason::StreamEnded,
            )
            .await;
            return Err(ListenWaitError::Tool(ToolResult::error(format!(
                "Listen stream timed out after {}s waiting for updates",
                LISTEN_UPDATE_TIMEOUT.as_secs()
            ))));
        };

        let Some(update) = next_update else {
            cancel_last_listen_operation(
                tool,
                tool_context,
                pending,
                last_operation_id.as_deref(),
                ListenStopReason::StreamEnded,
            )
            .await;
            return Err(ListenWaitError::Tool(ToolResult::error(
                "Listen stream ended before operation became ready",
            )));
        };

        update_count += 1;
        match handle_listen_update::<H>(
            pending,
            hooks,
            event_store,
            thread_id,
            turn,
            seq,
            update,
            &mut last_operation_id,
        )
        .await
        {
            Ok(ListenUpdateHandling::Continue) => {}
            Ok(ListenUpdateHandling::Ready(ready)) => return Ok(ready),
            Err(error) => return Err(error),
        }

        if update_count >= MAX_LISTEN_UPDATES {
            cancel_last_listen_operation(
                tool,
                tool_context,
                pending,
                last_operation_id.as_deref(),
                ListenStopReason::StreamEnded,
            )
            .await;
            return Err(ListenWaitError::Tool(ToolResult::error(format!(
                "Listen tool exceeded max updates ({MAX_LISTEN_UPDATES})"
            ))));
        }
    }
}
