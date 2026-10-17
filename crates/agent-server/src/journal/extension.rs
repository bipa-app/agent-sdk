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
//! For example, bip books a turn's cost into its cost document on every
//! completed turn, in the same transaction as the turn.

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
    /// This transition with `extension`'s writes appended.
    ///
    /// # Errors
    /// [`Rejected::Store`] when the extension returns a write it may not
    /// add; nothing is committed.
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
            crate::observability::attrs::VIEW_COMMIT,
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
    /// [`extra_writes`](Self::extra_writes).
    ///
    /// # Errors
    /// When the extension returns a write a completed turn may not carry;
    /// nothing is committed.
    pub fn extended(mut self, extension: Option<&dyn TransitionExtension>) -> anyhow::Result<Self> {
        let Some(extension) = extension else {
            return Ok(self);
        };
        let added = extension.extend(TransitionView::CompletedTurn(&self));
        let refused = added.iter().find(|write| !completed_turn_may_carry(write));
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_extension(
            crate::observability::attrs::VIEW_COMPLETED_TURN,
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
    /// transaction.
    ///
    /// # Errors
    /// Names the first write a completed turn may not carry, or the first
    /// malformed document.
    pub fn validate_extra_writes(&self) -> anyhow::Result<()> {
        for write in &self.extra_writes {
            anyhow::ensure!(
                completed_turn_may_carry(write),
                "a completed turn may carry only documents among its extra writes, not {write:?}"
            );
            if let Write::Document(document) = write {
                document.validate()?;
            }
        }
        Ok(())
    }

    /// The extra writes as a transition with no guard, for the backends'
    /// shared write arms. A completed turn checks its own ownership.
    #[must_use]
    pub fn extra_transition(&self) -> Transition {
        Transition {
            guards: Vec::new(),
            writes: self.extra_writes.clone(),
        }
    }
}
