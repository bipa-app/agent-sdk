//! The write set of one thread creation, committed by
//! [`super::thread_store::ThreadStore::commit_thread_creation`].

use agent_sdk_foundation::ThreadId;
use agent_sdk_foundation::events::AgentEvent;
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
