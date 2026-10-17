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
}

impl EpochRecord {
    /// A new epoch that starts at `now`.
    #[must_use]
    pub fn begin(now: OffsetDateTime, label: Option<String>) -> Self {
        Self {
            id: EpochId::new(),
            started_at: now,
            label,
        }
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
}

impl LeaseRecoveryScope {
    /// The scope's stable label, as in metrics: `expired` or
    /// `foreign_epoch`.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Expired => "expired",
            Self::ForeignEpoch(_) => "foreign_epoch",
        }
    }

    /// The recovery context the matrix classifies a reclaimed row in.
    #[must_use]
    pub const fn context(&self) -> super::recovery::RecoveryContext {
        match self {
            Self::Expired => super::recovery::RecoveryContext::ExpiredLease,
            Self::ForeignEpoch(_) => super::recovery::RecoveryContext::EpochChanged,
        }
    }

    /// Whether a `Running` row with this lease is reclaimed under the
    /// scope at `now`.
    #[must_use]
    pub fn reclaims(
        &self,
        lease_expires_at: Option<OffsetDateTime>,
        lease_epoch: Option<&EpochId>,
        now: OffsetDateTime,
    ) -> bool {
        match self {
            Self::Expired => lease_expires_at.is_some_and(|expires_at| expires_at <= now),
            Self::ForeignEpoch(current) => lease_epoch != Some(current),
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
        assert!(scope.reclaims(Some(t(1_000)), Some(&old), t(0)));
        assert!(scope.reclaims(Some(t(1_000)), None, t(0)));
        assert!(!scope.reclaims(Some(t(-1)), Some(&current), t(0)));
    }

    #[test]
    fn an_expired_scope_reclaims_by_expiry_only() {
        let epoch = EpochId::from_string("epoch-a");
        let scope = LeaseRecoveryScope::Expired;
        assert!(scope.reclaims(Some(t(0)), Some(&epoch), t(0)));
        assert!(!scope.reclaims(Some(t(1)), None, t(0)));
    }
}
