//! Transactional outbox for durable event relay.
//!
//! The outbox is a delivery buffer — not the authority for replay or
//! ordering.  `agent_sdk_committed_events` owns the canonical
//! thread-scoped sequence; the outbox merely holds relay work so an
//! AMQP or pub-sub worker can deliver notifications without weakening
//! the journal's commit guarantees.
//!
//! # Transactional semantics
//!
//! Outbox rows are inserted in the **same SQL transaction** as the
//! durable journal mutation that produced the events.  If the
//! transaction commits, the outbox rows exist; if it rolls back,
//! neither events nor outbox rows are visible.  This is the core
//! guarantee that makes the outbox pattern safe.
//!
//! # Relay lifecycle
//!
//! ```text
//!   Pending ──▶ Claimed ──▶ Delivered
//!                  │
//!                  └──▶ Failed ──▶ Pending  (retry, if budget allows)
//!                          │
//!                          └──▶ Expired     (max attempts exhausted)
//! ```
//!
//! Terminal durable state **never depends** on successful outbox relay.
//! A relay failure is an operational degradation (subscribers see stale
//! data) but not a correctness violation.
//!
//! # Idempotent consumption
//!
//! Consumers track the last-processed `(thread_id, sequence)` and
//! ignore duplicate deliveries.  The relay is at-least-once by design.

use agent_sdk_core::ThreadId;
use anyhow::{Result, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use time::OffsetDateTime;
use tokio::sync::RwLock;

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
    /// Relay attempt failed; row may be retried if budget allows.
    Failed,
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
            Self::Failed => "failed",
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

/// A single outbox row: a notification that an event needs relay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutboxRow {
    /// Unique outbox row identity.
    pub id: OutboxRowId,
    /// Thread the event belongs to.
    pub thread_id: ThreadId,
    /// The committed event's globally unique ID.
    pub event_id: uuid::Uuid,
    /// Copy of the event's thread-scoped sequence for ordering.
    pub sequence: u64,
    /// Relay lifecycle status.
    pub status: OutboxStatus,
    /// Self-contained relay payload (serialised event envelope).
    pub payload_json: serde_json::Value,
    /// When the outbox row was created (same transaction as event commit).
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
pub struct NewOutboxRow {
    pub thread_id: ThreadId,
    pub event_id: uuid::Uuid,
    pub sequence: u64,
    pub payload_json: serde_json::Value,
    pub max_attempts: u32,
    pub now: OffsetDateTime,
}

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Transactional outbox store for durable event relay.
///
/// Rows are inserted in the same SQL transaction as committed events.
/// Relay workers claim pending rows, attempt delivery, and mark them
/// as delivered or failed.
///
/// # Contract
///
/// - Rows are created only inside an event-commit transaction.
/// - `claim_pending` returns rows ordered by `next_attempt_at` so
///   older failures are retried before newer rows.
/// - `mark_delivered` and `mark_failed` are idempotent on terminal rows.
/// - Terminal durable state never depends on successful relay.
#[async_trait]
pub trait OutboxStore: Send + Sync {
    /// Insert one or more outbox rows.
    ///
    /// In the Postgres backend, this runs inside the same transaction
    /// as the corresponding `commit_event_batch`.
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

    /// Retrieve an outbox row by its ID.
    async fn get(&self, id: &OutboxRowId) -> Result<Option<OutboxRow>>;

    /// List outbox rows for a thread, ordered by sequence.
    async fn list_by_thread(&self, thread_id: &ThreadId) -> Result<Vec<OutboxRow>>;

    /// Count pending (undelivered, non-expired) rows for a thread.
    async fn count_pending(&self, thread_id: &ThreadId) -> Result<u64>;
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
            let row = OutboxRow {
                id: OutboxRowId::new(),
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

        claimable.sort_by(|a, b| a.next_attempt_at.cmp(&b.next_attempt_at));
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
        rows.sort_by_key(|r| r.sequence);
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
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_core::events::AgentEvent;
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-outbox-a")
    }

    fn sample_payload() -> serde_json::Value {
        serde_json::to_value(AgentEvent::text("msg_1", "hello")).unwrap_or_default()
    }

    fn sample_new_row(thread_id: &ThreadId, seq: u64, now: OffsetDateTime) -> NewOutboxRow {
        NewOutboxRow {
            thread_id: thread_id.clone(),
            event_id: uuid::Uuid::now_v7(),
            sequence: seq,
            payload_json: sample_payload(),
            max_attempts: 3,
            now,
        }
    }

    #[tokio::test]
    async fn insert_batch_creates_pending_rows() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![
                sample_new_row(&thread_a(), 0, t0()),
                sample_new_row(&thread_a(), 1, t0()),
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
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())])
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
                sample_new_row(&thread_a(), 0, t0()),
                sample_new_row(&thread_a(), 1, t0()),
                sample_new_row(&thread_a(), 2, t0()),
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
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())])
            .await?;
        let id = &rows[0].id;

        let claimed = store.claim_pending("worker-1", 10, t_plus(1)).await?;
        assert_eq!(claimed.len(), 1);

        store.mark_delivered(id, t_plus(2)).await?;

        let row = store.get(id).await?.expect("row should exist");
        assert_eq!(row.status, OutboxStatus::Delivered);
        assert!(row.delivered_at.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn mark_failed_retries_within_budget() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())])
            .await?;
        let id = &rows[0].id;

        store.claim_pending("worker-1", 10, t_plus(1)).await?;
        store
            .mark_failed(id, "connection refused", t_plus(60), t_plus(2))
            .await?;

        let row = store.get(id).await?.expect("row should exist");
        assert_eq!(row.status, OutboxStatus::Pending);
        assert_eq!(row.attempt_count, 1);
        // Pending rows must have NULL last_error per the outbox_error_check constraint.
        assert!(row.last_error.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn mark_failed_expires_when_budget_exhausted() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let mut params = sample_new_row(&thread_a(), 0, t0());
        params.max_attempts = 1;
        let rows = store.insert_batch(vec![params]).await?;
        let id = &rows[0].id;

        store.claim_pending("worker-1", 10, t_plus(1)).await?;
        store
            .mark_failed(id, "timeout", t_plus(60), t_plus(2))
            .await?;

        let row = store.get(id).await?.expect("row should exist");
        assert_eq!(row.status, OutboxStatus::Expired);
        assert!(row.status.is_terminal());
        Ok(())
    }

    #[tokio::test]
    async fn mark_delivered_is_idempotent_on_terminal() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        let rows = store
            .insert_batch(vec![sample_new_row(&thread_a(), 0, t0())])
            .await?;
        let id = &rows[0].id;

        store.claim_pending("worker-1", 10, t_plus(1)).await?;
        store.mark_delivered(id, t_plus(2)).await?;
        store.mark_delivered(id, t_plus(3)).await?;

        let row = store.get(id).await?.expect("row should exist");
        assert_eq!(row.status, OutboxStatus::Delivered);
        Ok(())
    }

    #[tokio::test]
    async fn count_pending_tracks_undelivered() -> Result<()> {
        let store = InMemoryOutboxStore::new();
        store
            .insert_batch(vec![
                sample_new_row(&thread_a(), 0, t0()),
                sample_new_row(&thread_a(), 1, t0()),
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
                sample_new_row(&thread_a(), 2, t0()),
                sample_new_row(&thread_a(), 0, t0()),
                sample_new_row(&thread_a(), 1, t0()),
            ])
            .await?;

        let rows = store.list_by_thread(&thread_a()).await?;
        let seqs: Vec<u64> = rows.iter().map(|r| r.sequence).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
        Ok(())
    }
}
