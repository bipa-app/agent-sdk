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
use super::document::DocScope;
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
        | Write::ReparkAfterSteering { .. }
        | Write::Effect(_)
        | Write::AppendCompaction { .. }
        | Write::SpawnChildren { .. } => false,
    }
}

/// Whether a completed turn may carry `write` among its extra writes.
pub(crate) const fn completed_turn_may_carry(write: &Write) -> bool {
    match write {
        Write::Document(_) => true,
        Write::SettleTask { .. }
        | Write::RejectConfirmation { .. }
        | Write::PauseOnConfirmation { .. }
        | Write::ReparkAfterSteering { .. }
        | Write::Effect(_)
        | Write::AppendCompaction { .. }
        | Write::AppendEvents { .. }
        | Write::SpawnChildren { .. } => false,
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
    /// When the extension returns a write a completed turn may not carry;
    /// nothing is committed.
    #[doc(hidden)]
    pub fn extended(mut self, extension: Option<&dyn TransitionExtension>) -> anyhow::Result<Self> {
        let Some(extension) = extension else {
            return Ok(self);
        };
        let added = extension.extend(TransitionView::CompletedTurn(&self));
        let refused = added.iter().find(|write| !completed_turn_may_carry(write));
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_extension(
            crate::observability::attrs::EXTENSION_POINT_COMPLETED_TURN,
            added.len(),
            refused.is_some(),
        );
        if let Some(refused) = refused {
            anyhow::bail!(
                "a transition extension may add only documents to a completed turn, not {refused:?}"
            );
        }
        self.extra_writes.extend(added);
        Ok(self)
    }

    /// Check the extra writes every backend applies inside the turn's
    /// transaction: documents only, each scoped to the turn's own thread or
    /// its own task.
    ///
    /// The scope rule keeps the purge fence and the lock order without a
    /// lock pass of their own: every completed turn locks its thread row
    /// first and refuses a thread that is being purged, and a document on
    /// that thread or on its task is reached only under that lock, as a
    /// transition reaches it under `Lock::Thread` / `Lock::ThreadOf`.
    ///
    /// # Errors
    /// Names the first write a completed turn may not carry, the first
    /// document outside the turn's own scope, or the first malformed
    /// document.
    #[doc(hidden)]
    pub fn validate_extra_writes(&self) -> anyhow::Result<()> {
        for write in &self.extra_writes {
            anyhow::ensure!(
                completed_turn_may_carry(write),
                "a completed turn may carry only documents among its extra writes, not {write:?}"
            );
            if let Write::Document(document) = write {
                let own = match &document.key.scope {
                    DocScope::Thread(thread) => thread == &self.thread_id,
                    DocScope::Task(task) => task == &self.task_id,
                    DocScope::Host => false,
                };
                anyhow::ensure!(
                    own,
                    "a completed turn may write documents only on its own thread {} or task {}, \
                     not {}",
                    self.thread_id,
                    self.task_id,
                    document.key
                );
                document.validate()?;
            }
        }
        Ok(())
    }

    /// The extra writes as a transition with no guard, for the backends'
    /// shared write arms.
    ///
    /// Reconciliation C2 sketches a `TaskIs` guard on the turn's task here.
    /// Documents need none: their scope is the turn's own thread or task
    /// (see [`Self::validate_extra_writes`]), which the turn locks first.
    /// The turn's ownership is checked by the turn itself only when
    /// [`owner_guard`](Self::owner_guard) is set. The PR that adds the
    /// root's settle to the extra writes (P2-3) adds the guard with it.
    #[doc(hidden)]
    #[must_use]
    pub fn extra_transition(&self) -> Transition {
        Transition {
            guards: Vec::new(),
            writes: self.extra_writes.clone(),
        }
    }
}
