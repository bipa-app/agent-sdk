//! The transition extension hook (design §6.4).
//!
//! A host registers one [`TransitionExtension`] on its store. The store
//! calls it once per [`DurableStore::commit`](super::DurableStore::commit)
//! and once per completed-turn commit, before the transaction opens, and
//! applies the writes it returns in the same transaction as the rest. The
//! extension is a pure function of what is being committed: it reads
//! nothing and does no I/O, so a retried transaction never calls it again
//! and its writes are as atomic as the writes it extends.
//!
//! For example, bip books a turn's cost into a document on the turn's own
//! thread on every completed turn, in the same transaction as the turn.
//!
//! The hook runs only on these two paths. The older single-purpose store
//! methods (`complete_task`, `cancel_tree`, `pause_on_question`, …) do not
//! call it; a host whose rule keys on a write must make sure that write
//! goes through `commit`.

use super::commit::CompletedTurnCommit;
use super::transition::{Rejected, Transition, Write};

/// What an extension sees.
#[derive(Clone, Copy, Debug)]
pub enum TransitionView<'a> {
    /// A transition on its way to [`super::DurableStore::commit`]. The
    /// extension may add [`Write::Document`] and [`Write::AppendEvents`].
    Commit(&'a Transition),
    /// A completed turn on its way to
    /// [`super::AgentTaskStore::commit_completed_turn`]. The extension may
    /// add [`Write::Document`] only.
    CompletedTurn(&'a CompletedTurnCommit),
}

/// Adds writes to what a store commits. See the module docs.
pub trait TransitionExtension: Send + Sync {
    /// The writes to add to `view`, applied after its own writes. Empty
    /// when the extension has nothing to add.
    fn extend(&self, view: TransitionView<'_>) -> Vec<Write>;
}

/// Whether an extension may add `write` to a transition.
const fn commit_may_add(write: &Write) -> bool {
    match write {
        Write::Document(_) | Write::AppendEvents { .. } => true,
        Write::SettleTask { .. }
        | Write::RejectConfirmation { .. }
        | Write::PauseOnConfirmation { .. }
        | Write::ParkInvocation { .. }
        | Write::WakeInvocation { .. }
        | Write::ReparkAfterSteering { .. }
        | Write::Effect(_)
        | Write::AppendCompaction { .. }
        | Write::SpawnChildren { .. }
        | Write::EnqueueSteeringResume { .. }
        | Write::BeginEpoch(_)
        | Write::TurnAttempt(_) => false,
    }
}

impl Transition {
    /// This transition with `extension`'s writes appended. The store calls
    /// this once per commit; a host never does.
    ///
    /// # Errors
    /// [`Rejected::Store`] when the extension returns a write it may not
    /// add; nothing is committed.
    #[doc(hidden)]
    pub fn extended(
        mut self,
        extension: Option<&dyn TransitionExtension>,
    ) -> Result<Self, Rejected> {
        let Some(extension) = extension else {
            return Ok(self);
        };
        let added = extension.extend(TransitionView::Commit(&self));
        let refused = added.iter().find(|write| !commit_may_add(write));
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_extension(
            crate::observability::attrs::EXTENSION_POINT_COMMIT,
            added.len(),
            refused.is_some(),
        );
        if let Some(refused) = refused {
            return Err(Rejected::Store(anyhow::anyhow!(
                "a transition extension may add only documents and events, not {refused:?}"
            )));
        }
        self.writes.extend(added);
        Ok(self)
    }
}

impl CompletedTurnCommit {
    /// This commit with `extension`'s writes appended to its
    /// [`extra_writes`](Self::extra_writes). The store calls this once per
    /// commit; a host never does.
    ///
    /// # Errors
    /// When the extension returns a write other than a document on the
    /// turn's own thread or task; nothing is committed, and the refusal is
    /// counted.
    #[doc(hidden)]
    pub fn extended(mut self, extension: Option<&dyn TransitionExtension>) -> anyhow::Result<Self> {
        let Some(extension) = extension else {
            return Ok(self);
        };
        let added = extension.extend(TransitionView::CompletedTurn(&self));
        let refused = added
            .iter()
            .find_map(|write| self.check_extension_write(write).err());
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_extension(
            crate::observability::attrs::EXTENSION_POINT_COMPLETED_TURN,
            added.len(),
            refused.is_some(),
        );
        if let Some(refused) = refused {
            return Err(refused);
        }
        self.extra_writes.extend(added);
        Ok(self)
    }

    /// Check a write an extension adds to this completed turn: only a
    /// document, on the turn's own thread or task.
    fn check_extension_write(&self, write: &Write) -> anyhow::Result<()> {
        match write {
            Write::Document(document) => self.check_own_document(document),
            Write::SettleTask { .. }
            | Write::RejectConfirmation { .. }
            | Write::PauseOnConfirmation { .. }
            | Write::ParkInvocation { .. }
            | Write::WakeInvocation { .. }
            | Write::ReparkAfterSteering { .. }
            | Write::Effect(_)
            | Write::AppendCompaction { .. }
            | Write::AppendEvents { .. }
            | Write::SpawnChildren { .. }
            | Write::EnqueueSteeringResume { .. }
            | Write::BeginEpoch(_)
            | Write::TurnAttempt(_) => anyhow::bail!(
                "a transition extension may add only documents to a completed turn, not {write:?}"
            ),
        }
    }
}
