//! The purge fence's refusals of writes outside a transition, as signals.

use agent_server::journal::thread::{FencedWriter, ThreadStateConflict, record_fence_refusal};

/// Report `refused` when it is the fence's [`ThreadStateConflict`]: an
/// `agent_server.fence.refused` event on the current span with the thread
/// id, plus the `agent_server.fence.refused` count. Returns `refused`.
///
/// # Errors
/// `refused` itself.
pub fn fence_refused(writer: FencedWriter, refused: anyhow::Result<()>) -> anyhow::Result<()> {
    if let Err(error) = &refused
        && let Some(conflict) = error.downcast_ref::<ThreadStateConflict>()
    {
        tracing::event!(
            name: "agent_server.fence.refused",
            tracing::Level::INFO,
            thread.id = conflict.thread_id.0.as_str(),
            writer = writer.as_str(),
            thread_status = conflict.status.as_str(),
            "the purge fence refused a write",
        );
    }
    record_fence_refusal(writer, refused)
}
