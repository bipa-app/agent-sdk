//! Atomic thread-creation commit hook for durable backends.

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
use anyhow::Result;
use async_trait::async_trait;
use time::OffsetDateTime;

use super::committed_event::CommittedEvent;
use super::thread::Thread;

/// Complete write set for creating one journal-visible thread.
pub struct ThreadCreationCommit {
    pub thread_id: ThreadId,
    pub event: AgentEvent,
    pub outbox_max_attempts: u32,
    pub now: OffsetDateTime,
}

/// Rows returned from a successful atomic thread creation.
pub struct ThreadCreationRows {
    pub thread: Thread,
    pub committed_event: CommittedEvent,
}

/// Backend hook that creates the thread aggregate, empty message projection,
/// creation event, and coalesced outbox advisory in one transaction.
#[async_trait]
pub trait AtomicThreadCreationCommitter: Send + Sync {
    async fn commit_thread_creation_atomic(
        &self,
        params: ThreadCreationCommit,
    ) -> Result<ThreadCreationRows>;
}
