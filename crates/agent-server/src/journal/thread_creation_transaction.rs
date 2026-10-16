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
    pub naming: ThreadNaming,
    pub event: AgentEvent,
    pub outbox_max_attempts: u32,
    pub now: OffsetDateTime,
}

/// Who chose a created thread's id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadNaming {
    /// The service minted a fresh id; nothing else can hold it.
    Minted,
    /// The caller named the id. The creation claims it under
    /// [`ThreadCreation::Create`](super::thread_store::ThreadCreation::Create)
    /// in the same unit, so a retry of a creation that landed returns
    /// [`ThreadCreationCommitted::Existing`] and writes nothing, and an id
    /// claimed differently is refused with
    /// [`ThreadIdConflict`](super::thread_store::ThreadIdConflict).
    CallerNamed,
}

/// Rows returned from a successful atomic thread creation.
pub struct ThreadCreationRows {
    pub thread: Thread,
    pub committed_event: CommittedEvent,
}

/// What [`super::thread_store::ThreadStore::commit_thread_creation`] did.
pub enum ThreadCreationCommitted {
    /// This call created the thread and committed its event.
    Created(Box<ThreadCreationRows>),
    /// A caller-named thread this creation already made; nothing written.
    Existing,
}
