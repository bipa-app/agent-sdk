//! Process epochs: proof of which host process holds a lease.
//!
//! A host writes a new [`EpochRecord`] in one transaction when it starts
//! ([`super::transition::Write::BeginEpoch`]), and every lease it grants
//! from then on carries that epoch. Worker ids repeat across restarts
//! (`worker-0..N`), so the epoch, not the worker, tells a dead process's
//! lease from a live one.
//!
//! A host that is the only process on its store (an embedder that holds a
//! single-instance lock, such as a desktop daemon) can then requeue every
//! lease of an older epoch at start, through
//! [`super::AgentTaskStore::release_expired_leases`] with
//! [`LeaseRecoveryScope::ForeignEpoch`], instead of waiting for the leases
//! to expire. Hosts that share a store (several processes on one Postgres
//! database, or overlapping pods during a rolling deploy) keep lease
//! expiry: another live process's leases carry another epoch too.
//!
//! Every lease grant stamps the epoch (`try_acquire_task`,
//! `acquire_ready_head`, `approve_confirmation_and_acquire`). Every lease
//! granted before the current epoch began is foreign whether or not it was
//! stamped, so the stamp only protects a lease granted between `BeginEpoch`
//! and the start-up reclaim; a host must not serve writes in that window
//! anyway. Beyond that it is operator data: which process granted a lease.
//! `agent_sdk_host_epochs` gains one row per process start and is never
//! pruned; at one row per start it stays small.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// The id of one host process's lifetime on a store.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EpochId(pub String);

impl EpochId {
    /// A fresh, unique epoch id.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("epoch_{}", uuid::Uuid::new_v4()))
    }

    /// Wrap an existing epoch id.
    #[must_use]
    pub fn from_string(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for EpochId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EpochId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The durable record of an epoch's start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochRecord {
    pub id: EpochId,
    pub started_at: OffsetDateTime,
    /// A free-form name for the process, for operators (for example the
    /// host name). Never read by recovery.
    pub label: Option<String>,
    /// The recovery mode the process began under, by its stable label
    /// (`lease_expiry`, `single_host`, `postgres_leader`). `None` for an
    /// epoch that records none. An epoch whose mode is in
    /// [`EXCLUSIVE_RECOVERY`] supersedes every other such epoch when it
    /// begins (see [`EpochRecord::is_exclusive`]).
    pub recovery: Option<String>,
}

/// The recovery modes whose process is the only one with workers on its
/// store.
///
/// When an epoch in one of them begins, the store marks every other epoch
/// in one of them superseded in the same transaction, and from then on
/// refuses that epoch's acquires, heartbeats and lease-guarded commits.
pub const EXCLUSIVE_RECOVERY: [&str; 2] = ["single_host", "postgres_leader"];

impl EpochRecord {
    /// A new epoch that starts at `now`.
    #[must_use]
    pub fn begin(now: OffsetDateTime, label: Option<String>) -> Self {
        Self {
            id: EpochId::new(),
            started_at: now,
            label,
            recovery: None,
        }
    }

    /// Record the recovery mode the epoch begins under, by its stable label.
    #[must_use]
    pub fn with_recovery(mut self, recovery: &str) -> Self {
        self.recovery = Some(recovery.to_owned());
        self
    }

    /// Whether the epoch begins under an exclusive mode, so that it
    /// supersedes the other exclusive epochs of its store.
    #[must_use]
    pub fn is_exclusive(&self) -> bool {
        self.recovery
            .as_deref()
            .is_some_and(|recovery| EXCLUSIVE_RECOVERY.contains(&recovery))
    }
}

/// A newer exclusive epoch superseded the epoch a write belongs to.
///
/// The process that holds that epoch no longer leads its store, and only
/// its successor may grant, extend or settle leases.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct EpochSuperseded(String);

impl EpochSuperseded {
    /// This store's own epoch was superseded, so it grants no lease.
    #[must_use]
    pub fn new(epoch: &EpochId) -> Self {
        Self(format!(
            "epoch {epoch} was superseded by a newer exclusive epoch; this process may not grant leases"
        ))
    }

    /// `task`'s lease was granted under a superseded epoch.
    #[must_use]
    pub fn of_lease(task: &super::task::AgentTaskId) -> Self {
        Self(format!(
            "the lease of task {task} belongs to an epoch a newer exclusive epoch superseded"
        ))
    }
}

/// Which `Running` rows [`super::AgentTaskStore::release_expired_leases`]
/// reclaims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseRecoveryScope {
    /// Rows whose lease has expired by `now`.
    Expired,
    /// Rows whose lease was granted under any epoch but this one,
    /// including rows leased before epochs existed, whatever their
    /// expiry. Only a host that is the sole process on its store may use
    /// it.
    ForeignEpoch(EpochId),
    /// Rows whose lease was granted under an epoch a newer exclusive epoch
    /// superseded, whatever their expiry. Leases of non-exclusive epochs
    /// (and of no epoch) wait for expiry, so a host that may share its
    /// store with hosts of another mode, as in a switch-over deploy, uses
    /// it.
    SupersededEpochs,
}

impl LeaseRecoveryScope {
    /// The scope's stable label, as in metrics: `expired` or
    /// `foreign_epoch`.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Expired => "expired",
            Self::ForeignEpoch(_) => "foreign_epoch",
            Self::SupersededEpochs => "superseded_epoch",
        }
    }

    /// The recovery context the matrix classifies a reclaimed row in.
    #[must_use]
    pub const fn context(&self) -> super::recovery::RecoveryContext {
        match self {
            Self::Expired => super::recovery::RecoveryContext::ExpiredLease,
            Self::ForeignEpoch(_) | Self::SupersededEpochs => {
                super::recovery::RecoveryContext::EpochChanged
            }
        }
    }

    /// Whether a `Running` row with this lease is reclaimed under the
    /// scope at `now`, given which epochs are superseded.
    #[must_use]
    pub fn reclaims(
        &self,
        lease_expires_at: Option<OffsetDateTime>,
        lease_epoch: Option<&EpochId>,
        superseded: impl Fn(&EpochId) -> bool,
        now: OffsetDateTime,
    ) -> bool {
        match self {
            Self::Expired => lease_expires_at.is_some_and(|expires_at| expires_at <= now),
            Self::ForeignEpoch(current) => lease_epoch != Some(current),
            Self::SupersededEpochs => lease_epoch.is_some_and(superseded),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(secs: i64) -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(secs)
    }

    #[test]
    fn a_foreign_epoch_scope_reclaims_every_lease_but_its_own() {
        let current = EpochId::from_string("epoch-b");
        let scope = LeaseRecoveryScope::ForeignEpoch(current.clone());
        let old = EpochId::from_string("epoch-a");
        assert!(scope.reclaims(Some(t(1_000)), Some(&old), |_| false, t(0)));
        assert!(scope.reclaims(Some(t(1_000)), None, |_| false, t(0)));
        assert!(!scope.reclaims(Some(t(-1)), Some(&current), |_| false, t(0)));
    }

    #[test]
    fn a_superseded_scope_reclaims_only_superseded_epochs() {
        let old = EpochId::from_string("epoch-a");
        let shared = EpochId::from_string("epoch-shared");
        let superseded = |epoch: &EpochId| epoch == &old;
        let scope = LeaseRecoveryScope::SupersededEpochs;
        assert!(scope.reclaims(Some(t(1_000)), Some(&old), superseded, t(0)));
        assert!(!scope.reclaims(Some(t(1_000)), Some(&shared), superseded, t(0)));
        assert!(!scope.reclaims(Some(t(1_000)), None, superseded, t(0)));
    }

    #[test]
    fn only_an_exclusive_mode_makes_an_exclusive_epoch() {
        let at = t(0);
        assert!(
            EpochRecord::begin(at, None)
                .with_recovery("postgres_leader")
                .is_exclusive()
        );
        assert!(
            EpochRecord::begin(at, None)
                .with_recovery("single_host")
                .is_exclusive()
        );
        assert!(
            !EpochRecord::begin(at, None)
                .with_recovery("lease_expiry")
                .is_exclusive()
        );
        assert!(!EpochRecord::begin(at, None).is_exclusive());
    }

    #[test]
    fn an_expired_scope_reclaims_by_expiry_only() {
        let epoch = EpochId::from_string("epoch-a");
        let scope = LeaseRecoveryScope::Expired;
        assert!(scope.reclaims(Some(t(0)), Some(&epoch), |_| false, t(0)));
        assert!(!scope.reclaims(Some(t(1)), None, |_| false, t(0)));
    }
}
