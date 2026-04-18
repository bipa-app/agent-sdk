//! Transactional outbox for durable broker relay.
//!
//! The outbox is a delivery buffer — never the authority for replay or
//! ordering.  `agent_sdk_committed_events` owns canonical thread-scoped
//! event order; `agent_sdk_tasks` owns canonical task state.  The
//! outbox merely holds *advisory* relay work so an AMQP or pub-sub
//! worker can publish hints without weakening the journal's commit
//! guarantees.
//!
//! # Transactional semantics
//!
//! Outbox rows are inserted in the **same SQL transaction** as the
//! durable journal mutation that produced them.  If the transaction
//! commits, the matching outbox row exists; if it rolls back, neither
//! the journal mutation nor the outbox row is visible.  This is the
//! core guarantee that makes the outbox pattern safe.
//!
//! # Logical kinds (Phase 8.1)
//!
//! Every row carries a [`OutboxMessageKind`] that tells the relay
//! worker what shape the payload has and which downstream subscribers
//! care about it.  See [`super::outbox_message`] for the full kind
//! enumeration and the advisory-payload contract.
//!
//! # Coalescing
//!
//! - `thread_events_available` rows are coalesced **per commit batch**
//!   — one row per `commit_events_with_outbox` call regardless of how
//!   many events landed.  The payload's `last_sequence` is the highest
//!   sequence the consumer is guaranteed to be able to read.
//! - `task_wakeup` rows are emitted **per task transition**, since
//!   each task lookup is independent on the consumer side.
//!
//! # Relay lifecycle
//!
//! ```text
//!   Pending ──▶ Claimed ──▶ Delivered
//!                  │
//!                  ├──▶ Pending  (retry, if budget allows)
//!                  │
//!                  └──▶ Expired  (max attempts exhausted)
//! ```
//!
//! Terminal durable state **never depends** on successful outbox relay.
//! A relay failure is an operational degradation (subscribers see stale
//! data) but not a correctness violation.
//!
//! # Idempotent consumption
//!
//! Consumers track their own per-thread cursor and ignore duplicate
//! deliveries.  The relay is at-least-once by design.

use agent_sdk_core::ThreadId;
use anyhow::{Result, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use time::{Duration, OffsetDateTime};
use tokio::sync::RwLock;

use super::outbox_message::OutboxMessageKind;

// ─────────────────────────────────────────────────────────────────────
// Identity
// ─────────────────────────────────────────────────────────────────────

/// Unique identifier for an outbox row.
///
/// Formatted as `outbox_<uuid>` to distinguish from event IDs,
/// task IDs, and other identity types in logs and audit trails.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OutboxRowId(pub String);

impl OutboxRowId {
    #[must_use]
    pub fn new() -> Self {
        Self(format!("outbox_{}", uuid::Uuid::new_v4()))
    }

    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for OutboxRowId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for OutboxRowId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Status
// ─────────────────────────────────────────────────────────────────────

/// Relay lifecycle status for an outbox row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutboxStatus {
    /// Awaiting relay pickup.
    Pending,
    /// Claimed by a relay worker.
    Claimed,
    /// Successfully delivered to the downstream consumer.
    Delivered,
    /// Max relay attempts exhausted; row is terminal.
    Expired,
}

impl OutboxStatus {
    /// True when the row is in a terminal state (delivered or expired).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Delivered | Self::Expired)
    }

    /// Wire-format string for durable persistence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Claimed => "claimed",
            Self::Delivered => "delivered",
            Self::Expired => "expired",
        }
    }
}

impl std::fmt::Display for OutboxStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Row
// ─────────────────────────────────────────────────────────────────────

/// A single outbox row: an advisory notification awaiting relay.
///
/// `event_id` and `sequence` are populated for
/// [`OutboxMessageKind::ThreadEventsAvailable`] rows, which reference
/// the LOWEST committed event in the triggering batch.  Using the
/// first (not last) event makes [`OutboxStore::min_unpublished_sequence`]
/// a correct retention-floor safety bound for every event in the
/// batch, including multi-event commits.  The advisory payload
/// separately carries `last_sequence` so subscribers know how far to
/// replay.  Both fields are `None` for
/// [`OutboxMessageKind::TaskWakeup`] rows, which reference only the
/// task / thread carried in `payload_json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutboxRow {
    /// Unique outbox row identity.
    pub id: OutboxRowId,
    /// Logical kind of the message this row carries.
    pub kind: OutboxMessageKind,
    /// Thread the message refers to.
    pub thread_id: ThreadId,
    /// Lowest committed event in the triggering batch (only set for
    /// `ThreadEventsAvailable` rows).
    #[serde(default)]
    pub event_id: Option<uuid::Uuid>,
    /// Lowest committed sequence in the triggering batch (only set
    /// for `ThreadEventsAvailable` rows).  Acts as the retention
    /// safety bound over the entire batch range.
    #[serde(default)]
    pub sequence: Option<u64>,
    /// Relay lifecycle status.
    pub status: OutboxStatus,
    /// Advisory payload: durable references only, no body data.
    ///
    /// Always shaped as the matching `*Payload` struct in
    /// [`super::outbox_message`].  Use
    /// [`OutboxMessage::from_payload_json`](super::outbox_message::OutboxMessage::from_payload_json)
    /// to decode.
    pub payload_json: serde_json::Value,
    /// When the outbox row was created (same transaction as the
    /// triggering journal mutation).
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// When the relay should next attempt delivery.
    #[serde(with = "time::serde::rfc3339")]
    pub next_attempt_at: OffsetDateTime,
    /// Number of relay attempts so far.
    pub attempt_count: u32,
    /// Maximum relay attempts before the row expires.
    pub max_attempts: u32,
    /// Most recent relay error, if any.
    pub last_error: Option<String>,
    /// Relay worker identity that claimed this row.
    pub claimed_by: Option<String>,
    /// When the relay worker claimed this row.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub claimed_at: Option<OffsetDateTime>,
    /// When the relay successfully delivered this row.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub delivered_at: Option<OffsetDateTime>,
}

/// Parameters for inserting a new outbox row.
///
/// For `ThreadEventsAvailable` rows, `event_id` and `sequence` MUST be
/// `Some` and refer to the LOWEST committed event in the triggering
/// batch (the safety bound for the retention janitor).  For
/// `TaskWakeup` rows, both MUST be `None` — the transactional rule is
/// enforced at the database layer via a CHECK constraint.
pub struct NewOutboxRow {
    pub kind: OutboxMessageKind,
    pub thread_id: ThreadId,
    pub event_id: Option<uuid::Uuid>,
    pub sequence: Option<u64>,
    pub payload_json: serde_json::Value,
    pub max_attempts: u32,
    pub now: OffsetDateTime,
}

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Transactional outbox store for durable broker relay.
///
/// Rows are inserted in the same SQL transaction as the journal
/// mutation that produced them.  Relay workers claim pending rows,
/// attempt delivery, and mark them as delivered or failed.
///
/// # Contract
///
/// - Rows are created only inside a transaction that also writes the
///   durable state they advertise (committed events for
///   `ThreadEventsAvailable`, task-journal mutations for
///   `TaskWakeup`).
/// - `claim_pending` returns rows ordered by `next_attempt_at` so
///   older failures are retried before newer rows.
/// - `mark_delivered` and `mark_failed` are idempotent on terminal rows.
/// - Terminal durable state never depends on successful relay.
/// - Workers MUST NOT publish to a broker outside this contract; the
///   outbox is the only authoritative path.
#[async_trait]
pub trait OutboxStore: Send + Sync {
    /// Insert one or more outbox rows.
    ///
    /// In durable backends, callers are expected to invoke this
    /// inside the same SQL transaction as the journal mutation that
    /// advertised the row — see
    /// [`AtomicEventOutboxCommitter`](super::event_outbox_transaction::AtomicEventOutboxCommitter)
    /// for the events path.  The in-memory implementation accepts
    /// stand-alone calls so tests can construct fixtures.
    async fn insert_batch(&self, rows: Vec<NewOutboxRow>) -> Result<Vec<OutboxRow>>;

    /// Claim up to `limit` pending rows for relay, ordered by
    /// `next_attempt_at` ascending.
    ///
    /// Claimed rows transition from `Pending` → `Claimed` and record
    /// the `worker_id`.
    async fn claim_pending(
        &self,
        worker_id: &str,
        limit: u32,
        now: OffsetDateTime,
    ) -> Result<Vec<OutboxRow>>;

    /// Mark a claimed row as successfully delivered.
    async fn mark_delivered(&self, id: &OutboxRowId, now: OffsetDateTime) -> Result<()>;

    /// Mark a claimed row as failed.
    ///
    /// If the row's retry budget is exhausted, it transitions to
    /// `Expired` instead of returning to `Pending`.
    async fn mark_failed(
        &self,
        id: &OutboxRowId,
        error: &str,
        next_attempt_at: OffsetDateTime,
        now: OffsetDateTime,
    ) -> Result<()>;

    /// Return stale `Claimed` rows to `Pending` so they can be retried.
    ///
    /// A claim is "stale" when the worker that claimed it has been holding
    /// the row for longer than `claim_lease` — typically because the
    /// worker crashed between a successful broker publish and the
    /// matching `mark_delivered` write.  Pass `claim_lease = 0` only when
    /// the caller intentionally wants to reclaim every currently-claimed
    /// row unconditionally.
    ///
    /// Reclaim is intentionally *not* a failure: `attempt_count` is
    /// preserved, `last_error` stays NULL, and `next_attempt_at` is reset
    /// to `now` so the row is eligible for immediate re-pickup.  The
    /// broker already guarantees at-least-once; the possible duplicate
    /// republish is the whole point of the pattern.
    ///
    /// Implementations MUST NOT reclaim terminal rows (`Delivered`,
    /// `Expired`) and MUST return the number of rows actually reclaimed
    /// so callers can decide whether to log or escalate.
    async fn reclaim_expired_claims(
        &self,
        now: OffsetDateTime,
        claim_lease: Duration,
    ) -> Result<u64>;

    /// Retrieve an outbox row by its ID.
    async fn get(&self, id: &OutboxRowId) -> Result<Option<OutboxRow>>;

    /// List outbox rows for a thread, ordered by sequence.
    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<OutboxRow>>;

    /// Count pending (undelivered, non-expired) rows for a thread.
    async fn count_pending(&self, thread_id: &ThreadId) -> Result<u64>;

    /// Return the lowest `sequence` among non-terminal
    /// (`Pending` / `Claimed`) outbox rows for a thread.
    ///
    /// Returns `None` when no unpublished rows exist or when all
    /// non-terminal rows are `TaskWakeup` kind (which carry no
    /// sequence).  The retention janitor uses this bound to ensure it
    /// never advances the event retention floor past an unpublished
    /// outbox row.
    async fn min_unpublished_sequence(&self, thread_id: &ThreadId) -> Result<Option<u64>>;
}

// ─────────────────────────────────────────────────────────────────────
// Invariants
// ─────────────────────────────────────────────────────────────────────

/// Returns true iff the `(event_id, sequence)` pairing is consistent
/// with `kind`.
///
/// `ThreadEventsAvailable` rows MUST carry both references; `TaskWakeup`
/// rows MUST carry neither.  The `Postgres` / `SQLite` migrations enforce
/// the same invariant via CHECK constraints; this helper lets in-memory
/// stores and durable backends reject violations *before* hitting the
/// database, with a clearer error message.
#[must_use]
pub const fn kind_payload_invariants_hold(
    kind: OutboxMessageKind,
    event_id: Option<uuid::Uuid>,
    sequence: Option<u64>,
) -> bool {
    match kind {
        OutboxMessageKind::ThreadEventsAvailable => event_id.is_some() && sequence.is_some(),
        OutboxMessageKind::TaskWakeup => event_id.is_none() && sequence.is_none(),
    }
}

// ─────────────────────────────────────────────────────────────────────
// In-memory implementation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct InMemoryOutboxStoreInner {
    rows: HashMap<String, OutboxRow>,
}

/// In-memory reference implementation of [`OutboxStore`].
///
/// Cloning shares the same underlying outbox state.
#[derive(Clone, Default)]
pub struct InMemoryOutboxStore {
    inner: Arc<RwLock<InMemoryOutboxStoreInner>>,
}

impl InMemoryOutboxStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl OutboxStore for InMemoryOutboxStore {
    async fn insert_batch(&self, rows: Vec<NewOutboxRow>) -> Result<Vec<OutboxRow>> {
        ensure!(!rows.is_empty(), "cannot insert an empty outbox batch");

        let mut inner = self.inner.write().await;
        let mut result = Vec::with_capacity(rows.len());

        for params in rows {
            ensure!(
                kind_payload_invariants_hold(params.kind, params.event_id, params.sequence),
                "outbox row of kind {} has incompatible event_id/sequence",
                params.kind,
            );

            let row = OutboxRow {
                id: OutboxRowId::new(),
                kind: params.kind,
                thread_id: params.thread_id,
                event_id: params.event_id,
                sequence: params.sequence,
                status: OutboxStatus::Pending,
                payload_json: params.payload_json,
                created_at: params.now,
                next_attempt_at: params.now,
                attempt_count: 0,
                max_attempts: params.max_attempts,
                last_error: None,
                claimed_by: None,
                claimed_at: None,
                delivered_at: None,
            };
            inner.rows.insert(row.id.0.clone(), row.clone());
            result.push(row);
        }
        drop(inner);

        Ok(result)
    }

    async fn claim_pending(
        &self,
        worker_id: &str,
        limit: u32,
        now: OffsetDateTime,
    ) -> Result<Vec<OutboxRow>> {
        let mut inner = self.inner.write().await;
        let mut claimable: Vec<&mut OutboxRow> = inner
            .rows
            .values_mut()
            .filter(|r| r.status == OutboxStatus::Pending && r.next_attempt_at <= now)
            .collect();

        claimable.sort_by_key(|a| a.next_attempt_at);
        claimable.truncate(limit as usize);

        let mut result = Vec::with_capacity(claimable.len());
        for row in claimable {
            row.status = OutboxStatus::Claimed;
            row.claimed_by = Some(worker_id.to_string());
            row.claimed_at = Some(now);
            result.push(row.clone());
        }
        drop(inner);

        Ok(result)
    }

    async fn mark_delivered(&self, id: &OutboxRowId, now: OffsetDateTime) -> Result<()> {
        let mut inner = self.inner.write().await;
        let row = inner
            .rows
            .get_mut(&id.0)
            .ok_or_else(|| anyhow::anyhow!("outbox row not found: {id}"))?;

        if row.status.is_terminal() {
            drop(inner);
            return Ok(());
        }

        row.status = OutboxStatus::Delivered;
        row.delivered_at = Some(now);
        drop(inner);
        Ok(())
    }

    async fn mark_failed(
        &self,
        id: &OutboxRowId,
        error: &str,
        next_attempt_at: OffsetDateTime,
        _now: OffsetDateTime,
    ) -> Result<()> {
        let mut inner = self.inner.write().await;
        let row = inner
            .rows
            .get_mut(&id.0)
            .ok_or_else(|| anyhow::anyhow!("outbox row not found: {id}"))?;

        if row.status.is_terminal() {
            drop(inner);
            return Ok(());
        }

        row.attempt_count += 1;

        if row.attempt_count >= row.max_attempts {
            row.status = OutboxStatus::Expired;
            row.last_error = Some(error.to_string());
        } else {
            row.status = OutboxStatus::Pending;
            row.last_error = None;
            row.next_attempt_at = next_attempt_at;
            row.claimed_by = None;
            row.claimed_at = None;
        }
        drop(inner);

        Ok(())
    }

    async fn reclaim_expired_claims(
        &self,
        now: OffsetDateTime,
        claim_lease: Duration,
    ) -> Result<u64> {
        let mut inner = self.inner.write().await;
        let mut reclaimed = 0u64;
        for row in inner.rows.values_mut() {
            if row.status != OutboxStatus::Claimed {
                continue;
            }
            // Treat missing claimed_at as "stale immediately" — the row
            // can only have reached `Claimed` through `claim_pending`,
            // which always stamps `claimed_at`, but being lenient here
            // keeps the reclaim path robust against any backend that
            // fills in the status differently (or rows rewritten by a
            // migration).
            let claimed_at = row.claimed_at.unwrap_or(OffsetDateTime::UNIX_EPOCH);
            if claimed_at + claim_lease > now {
                continue;
            }
            row.status = OutboxStatus::Pending;
            row.claimed_by = None;
            row.claimed_at = None;
            // Reset the retry clock to `now` so the row is immediately
            // eligible for re-pickup instead of waiting out the
            // original `next_attempt_at`.
            row.next_attempt_at = now;
            reclaimed += 1;
        }
        drop(inner);
        Ok(reclaimed)
    }

    async fn get(&self, id: &OutboxRowId) -> Result<Option<OutboxRow>> {
        let inner = self.inner.read().await;
        let result = inner.rows.get(&id.0).cloned();
        drop(inner);
        Ok(result)
    }

    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<OutboxRow>> {
        let inner = self.inner.read().await;
        let mut rows: Vec<OutboxRow> = inner
            .rows
            .values()
            .filter(|r| r.thread_id == *thread_id)
            .cloned()
            .collect();
        drop(inner);
        rows.sort_by_key(|r| (r.sequence.is_none(), r.sequence, r.id.clone()));
        Ok(rows)
    }

    async fn count_pending(&self, thread_id: &ThreadId) -> Result<u64> {
        let inner = self.inner.read().await;
        let count = inner
            .rows
            .values()
            .filter(|r| {
                r.thread_id == *thread_id
                    && matches!(r.status, OutboxStatus::Pending | OutboxStatus::Claimed)
            })
            .count();
        drop(inner);
        Ok(count as u64)
    }

    async fn min_unpublished_sequence(&self, thread_id: &ThreadId) -> Result<Option<u64>> {
        let inner = self.inner.read().await;
        let min_seq = inner
            .rows
            .values()
            .filter(|r| {
                r.thread_id == *thread_id
                    && matches!(r.status, OutboxStatus::Pending | OutboxStatus::Claimed)
                    && r.sequence.is_some()
            })
            .filter_map(|r| r.sequence)
            .min();
        drop(inner);
        Ok(min_seq)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::outbox_message::{
        OutboxMessage, OutboxMessageKind, TaskWakeupPayload, ThreadEventsAvailablePayload,
    };
    use crate::journal::task::AgentTaskId;
    use anyhow::Context;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-outbox-a")
    }

    fn thread_events_payload(
        thread_id: &ThreadId,
        last_sequence: u64,
    ) -> Result<serde_json::Value> {
        OutboxMessage::ThreadEventsAvailable(ThreadEventsAvailablePayload {
            thread_id: thread_id.clone(),
            last_sequence,
        })
        .to_payload_json()
        .context("serialise thread_events_available payload")
    }

    fn task_wakeup_payload(
        task_id: &AgentTaskId,
        thread_id: &ThreadId,
    ) -> Result<serde_json::Value> {
        OutboxMessage::TaskWakeup(TaskWakeupPayload {
            task_id: task_id.clone(),
            thread_id: thread_id.clone(),
        })
        .to_payload_json()
        .context("serialise task_wakeup payload")
    }

    fn sample_new_row(thread_id: &ThreadId, seq: u64, now: OffsetDateTime) -> Result<NewOutboxRow> {
        Ok(NewOutboxRow {
            kind: OutboxMessageKind::ThreadEventsAvailable,
            thread_id: thread_id.clone(),
            event_id: Some(uuid::Uuid::now_v7()),
            sequence: Some(seq),
            payload_json: thread_events_payload(thread_id, seq)?,
            max_attempts: 3,
            now,
        })
    }

    fn sample_task_wakeup_row(thread_id: &ThreadId, now: OffsetDateTime) -> Result<NewOutboxRow> {
        let task_id = AgentTaskId::new();
        Ok(NewOutboxRow {
            kind: OutboxMessageKind::TaskWakeup,
            thread_id: thread_id.clone(),
            event_id: None,
            sequence: None,
            payload_json: task_wakeup_payload(&task_id, thread_id)?,
            max_attempts: 3,
            now,
        })
    }

    #[tokio::test]
    async fn insert_batch_creates_pending_rows() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![
                sample_new_row(&thread_a(), 0, t0())?,
                sample_new_row(&thread_a(), 1, t0())?,
            ])
            .await?;

        assert_eq!(rows.len(), 2);
        for row in &rows {
            assert_eq!(row.status, OutboxStatus::Pending);
            assert_eq!(row.attempt_count, 0);
            assert!(row.claimed_by.is_none());
        }
        Ok(())
    }

    #[tokio::test]
    async fn empty_batch_is_rejected() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let result = store.insert_batch(vec![]).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn claim_pending_transitions_to_claimed() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;

        let claimed = store.claim_pending("worker-1", 10, t_plus(1)).await?;
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].status, OutboxStatus::Claimed);
        assert_eq!(claimed[0].claimed_by.as_deref(), Some("worker-1"));
        Ok(())
    }

    #[tokio::test]
    async fn claim_respects_limit() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        store
            .insert_batch(vec![
                sample_new_row(&thread_a(), 0, t0())?,
                sample_new_row(&thread_a(), 1, t0())?,
                sample_new_row(&thread_a(), 2, t0())?,
            ])
            .await?;

        let claimed = store.claim_pending("worker-1", 2, t_plus(1)).await?;
        assert_eq!(claimed.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn mark_delivered_transitions_to_terminal() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;
        let id = &rows[0].id;

        let claimed = store.claim_pending("worker-1", 10, t_plus(1)).await?;
        assert_eq!(claimed.len(), 1);

        store.mark_delivered(id, t_plus(2)).await?;

        let row = store.get(id).await?.context("row should exist")?;
        assert_eq!(row.status, OutboxStatus::Delivered);
        assert!(row.delivered_at.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn mark_failed_retries_within_budget() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;
        let id = &rows[0].id;

        store.claim_pending("worker-1", 10, t_plus(1)).await?;
        store
            .mark_failed(id, "connection refused", t_plus(60), t_plus(2))
            .await?;

        let row = store.get(id).await?.context("row should exist")?;
        assert_eq!(row.status, OutboxStatus::Pending);
        assert_eq!(row.attempt_count, 1);
        // Pending rows must have NULL last_error per the outbox_error_check constraint.
        assert!(row.last_error.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn mark_failed_expires_when_budget_exhausted() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let mut params = sample_new_row(&thread_a(), 0, t0())?;
        params.max_attempts = 1;
        let rows = store.insert_batch(vec![params]).await?;
        let id = &rows[0].id;

        store.claim_pending("worker-1", 10, t_plus(1)).await?;
        store
            .mark_failed(id, "timeout", t_plus(60), t_plus(2))
            .await?;

        let row = store.get(id).await?.context("row should exist")?;
        assert_eq!(row.status, OutboxStatus::Expired);
        assert!(row.status.is_terminal());
        Ok(())
    }

    #[tokio::test]
    async fn mark_delivered_is_idempotent_on_terminal() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;
        let id = &rows[0].id;

        store.claim_pending("worker-1", 10, t_plus(1)).await?;
        store.mark_delivered(id, t_plus(2)).await?;
        store.mark_delivered(id, t_plus(3)).await?;

        let row = store.get(id).await?.context("row should exist")?;
        assert_eq!(row.status, OutboxStatus::Delivered);
        Ok(())
    }

    #[tokio::test]
    async fn count_pending_tracks_undelivered() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        store
            .insert_batch(vec![
                sample_new_row(&thread_a(), 0, t0())?,
                sample_new_row(&thread_a(), 1, t0())?,
            ])
            .await?;

        assert_eq!(store.count_pending(&thread_a()).await?, 2);

        let claimed = store.claim_pending("worker-1", 1, t_plus(1)).await?;
        store.mark_delivered(&claimed[0].id, t_plus(2)).await?;

        assert_eq!(store.count_pending(&thread_a()).await?, 1);
        Ok(())
    }

    #[tokio::test]
    async fn list_by_thread_returns_in_sequence_order() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        store
            .insert_batch(vec![
                sample_new_row(&thread_a(), 2, t0())?,
                sample_new_row(&thread_a(), 0, t0())?,
                sample_new_row(&thread_a(), 1, t0())?,
            ])
            .await?;

        let rows = store.list_by_thread(&thread_a()).await?;
        let seqs: Vec<Option<u64>> = rows.iter().map(|r| r.sequence).collect();
        assert_eq!(seqs, vec![Some(0), Some(1), Some(2)]);
        Ok(())
    }

    // ── Phase 8.1 contract assertions ───────────────────────────────

    #[tokio::test]
    async fn task_wakeup_row_skips_event_references() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_task_wakeup_row(&thread_a(), t0())?])
            .await?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, OutboxMessageKind::TaskWakeup);
        assert!(rows[0].event_id.is_none());
        assert!(rows[0].sequence.is_none());

        // The advisory payload must round-trip back to the original.
        let message = OutboxMessage::from_payload_json(rows[0].kind, rows[0].payload_json.clone())?;
        let OutboxMessage::TaskWakeup(payload) = message else {
            panic!("expected TaskWakeup, got {message:?}");
        };
        assert_eq!(payload.thread_id, thread_a());
        Ok(())
    }

    #[tokio::test]
    async fn thread_events_row_carries_event_references() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 7, t0())?])
            .await?;

        assert_eq!(rows[0].kind, OutboxMessageKind::ThreadEventsAvailable);
        assert!(rows[0].event_id.is_some());
        assert_eq!(rows[0].sequence, Some(7));

        let message = OutboxMessage::from_payload_json(rows[0].kind, rows[0].payload_json.clone())?;
        let OutboxMessage::ThreadEventsAvailable(payload) = message else {
            panic!("expected ThreadEventsAvailable, got {message:?}");
        };
        assert_eq!(payload.thread_id, thread_a());
        assert_eq!(payload.last_sequence, 7);
        Ok(())
    }

    #[tokio::test]
    async fn invariant_rejects_thread_events_without_sequence() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let mut row = sample_new_row(&thread_a(), 0, t0())?;
        row.sequence = None;
        let result = store.insert_batch(vec![row]).await;
        assert!(
            result.is_err(),
            "ThreadEventsAvailable rows must carry sequence references",
        );
        Ok(())
    }

    #[tokio::test]
    async fn invariant_rejects_task_wakeup_with_event_references() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let mut row = sample_task_wakeup_row(&thread_a(), t0())?;
        row.event_id = Some(uuid::Uuid::now_v7());
        let result = store.insert_batch(vec![row]).await;
        assert!(
            result.is_err(),
            "TaskWakeup rows must NOT carry event_id references",
        );
        Ok(())
    }

    #[tokio::test]
    async fn list_by_thread_orders_task_wakeups_after_thread_events() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        store
            .insert_batch(vec![
                sample_new_row(&thread_a(), 0, t0())?,
                sample_task_wakeup_row(&thread_a(), t0())?,
                sample_new_row(&thread_a(), 1, t0())?,
            ])
            .await?;

        let rows = store.list_by_thread(&thread_a()).await?;
        let kinds: Vec<OutboxMessageKind> = rows.iter().map(|r| r.kind).collect();
        let seqs: Vec<Option<u64>> = rows.iter().map(|r| r.sequence).collect();
        assert_eq!(
            kinds,
            vec![
                OutboxMessageKind::ThreadEventsAvailable,
                OutboxMessageKind::ThreadEventsAvailable,
                OutboxMessageKind::TaskWakeup,
            ]
        );
        assert_eq!(seqs, vec![Some(0), Some(1), None]);
        Ok(())
    }

    // ── Phase 8.2: reclaim for crash recovery ───────────────────────

    #[tokio::test]
    async fn reclaim_resets_stale_claims_to_pending() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;
        let id = &rows[0].id;

        // Claim at t+1.  Lease is 10s; at t+30 the claim is stale.
        store.claim_pending("crashed-worker", 10, t_plus(1)).await?;
        let reclaimed = store
            .reclaim_expired_claims(t_plus(30), Duration::seconds(10))
            .await?;
        assert_eq!(reclaimed, 1);

        let row = store.get(id).await?.context("row should exist")?;
        assert_eq!(row.status, OutboxStatus::Pending);
        assert!(row.claimed_by.is_none());
        assert!(row.claimed_at.is_none());
        // Attempt count is preserved — reclaim is NOT a failure.
        assert_eq!(row.attempt_count, 0);
        // next_attempt_at is reset to now so the row is immediately pickable.
        assert_eq!(row.next_attempt_at, t_plus(30));
        Ok(())
    }

    #[tokio::test]
    async fn reclaim_with_zero_lease_reclaims_all_claimed_rows() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        store
            .insert_batch(vec![
                sample_new_row(&thread_a(), 0, t0())?,
                sample_new_row(&thread_a(), 1, t0())?,
                sample_new_row(&thread_a(), 2, t0())?,
            ])
            .await?;

        store.claim_pending("crashed-worker", 10, t_plus(1)).await?;
        let reclaimed = store
            .reclaim_expired_claims(t_plus(1), Duration::ZERO)
            .await?;
        assert_eq!(reclaimed, 3);

        // All rows should be re-claimable by a new worker.
        let rows = store.claim_pending("new-worker", 10, t_plus(2)).await?;
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert_eq!(row.claimed_by.as_deref(), Some("new-worker"));
        }
        Ok(())
    }

    #[tokio::test]
    async fn reclaim_skips_rows_whose_lease_has_not_expired() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;
        let id = &rows[0].id;

        store.claim_pending("live-worker", 10, t_plus(1)).await?;
        // At t+5 the lease (10s) is still live.
        let reclaimed = store
            .reclaim_expired_claims(t_plus(5), Duration::seconds(10))
            .await?;
        assert_eq!(reclaimed, 0);

        let row = store.get(id).await?.context("row should exist")?;
        assert_eq!(row.status, OutboxStatus::Claimed);
        assert_eq!(row.claimed_by.as_deref(), Some("live-worker"));
        Ok(())
    }

    #[tokio::test]
    async fn reclaim_skips_terminal_rows() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;
        let id = &rows[0].id;

        store.claim_pending("worker-1", 10, t_plus(1)).await?;
        store.mark_delivered(id, t_plus(2)).await?;

        let reclaimed = store
            .reclaim_expired_claims(t_plus(100), Duration::ZERO)
            .await?;
        assert_eq!(reclaimed, 0);

        let row = store.get(id).await?.context("row should exist")?;
        assert_eq!(row.status, OutboxStatus::Delivered);
        Ok(())
    }

    #[tokio::test]
    async fn reclaim_preserves_attempt_count_unlike_mark_failed() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;
        let id = &rows[0].id;

        // Simulate one failed delivery, then a crashed second attempt.
        store.claim_pending("worker-1", 10, t_plus(1)).await?;
        store
            .mark_failed(id, "first failure", t_plus(30), t_plus(2))
            .await?;
        // The first failure bumped attempt_count to 1; now claim again.
        store.claim_pending("worker-1", 10, t_plus(60)).await?;
        let mid = store.get(id).await?.context("row should exist")?;
        assert_eq!(mid.attempt_count, 1);
        assert_eq!(mid.status, OutboxStatus::Claimed);

        // Second attempt crashes — reclaim must not charge another attempt.
        let reclaimed = store
            .reclaim_expired_claims(t_plus(120), Duration::seconds(10))
            .await?;
        assert_eq!(reclaimed, 1);
        let after = store.get(id).await?.context("row should exist")?;
        assert_eq!(after.status, OutboxStatus::Pending);
        assert_eq!(after.attempt_count, 1, "reclaim must not charge an attempt");
        Ok(())
    }

    #[tokio::test]
    async fn payload_object_does_not_carry_kind_tag() -> Result<()> {
        // The kind lives in the outbox column; embedding it inside the
        // payload would invite consumers to derive routing from the
        // payload body, which is explicitly out of contract.
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())?])
            .await?;
        let payload = rows[0]
            .payload_json
            .as_object()
            .context("payload must be a JSON object")?;
        assert!(payload.get("kind").is_none());
        Ok(())
    }
}
