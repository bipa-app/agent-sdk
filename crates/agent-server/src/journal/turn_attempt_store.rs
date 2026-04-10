//! Append-only storage trait for [`TurnAttempt`] audit rows.
//!
//! The [`TurnAttemptStore`] trait is the sole write surface for
//! turn-attempt audit records. Its key design properties:
//!
//! 1. **Append-only apart from close** — new rows are inserted via
//!    [`TurnAttemptStore::open_attempt`] and the only mutation is
//!    [`TurnAttemptStore::close_attempt`]. There is no `update()` or
//!    `delete()`.
//! 2. **No continuation state** — the store records what happened
//!    during an LLM call, not what should happen next. Scheduler and
//!    continuation state live on [`super::task::AgentTask`] and
//!    [`super::task_state::TaskState`].
//! 3. **Full provenance** — every row carries the provider, requested
//!    model, response model, response id, and request/response blobs
//!    so the audit trail survives provider rotations.
//!
//! [`InMemoryTurnAttemptStore`] is the reference implementation,
//! following the same `Arc<RwLock<Inner>>` pattern as
//! [`super::thread_store::InMemoryThreadStore`].
//!
//! # Entry point summary
//!
//! | Entry point | What it does | Guard |
//! |-------------|-------------|-------|
//! | [`TurnAttemptStore::open_attempt`] | Inserts a new open row | — |
//! | [`TurnAttemptStore::close_attempt`] | Fills response fields on an open row | Must be open |
//! | [`TurnAttemptStore::get`] | Reads a single row by id | — |
//! | [`TurnAttemptStore::list_by_task`] | Lists all attempts for a task | Ordered by `attempt_number` |

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use time::OffsetDateTime;
use tokio::sync::RwLock;

use super::task::AgentTaskId;
use super::turn_attempt::{CloseAttemptParams, OpenAttemptParams, TurnAttempt, TurnAttemptId};

/// Storage trait for [`TurnAttempt`] audit rows.
///
/// The trait surface is deliberately narrow: `open_attempt` inserts,
/// `close_attempt` is the sole mutation, and reads are by id or
/// by task. There is no `update()` or `delete()` because the
/// audit table is append-only.
///
/// Implementations must guarantee that `close_attempt` on an
/// already-closed row returns an error, and that `list_by_task`
/// returns rows ordered by `attempt_number` ascending.
#[async_trait]
pub trait TurnAttemptStore: Send + Sync {
    /// Insert a new open attempt.
    ///
    /// The returned [`TurnAttempt`] has no response fields — it is
    /// ready for the LLM call. The caller should persist the returned
    /// row's `id` so it can close the attempt later.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be written.
    async fn open_attempt(&self, params: OpenAttemptParams) -> Result<TurnAttempt>;

    /// Close an open attempt with the LLM call's results.
    ///
    /// This is the only mutation the audit table supports. A
    /// successful call atomically fills in the response fields and
    /// marks the row as closed.
    ///
    /// # Errors
    /// - [`super::TurnAttemptSchemaError::AlreadyClosed`] if the row is
    ///   already closed.
    /// - `attempt not found` if the id does not exist.
    /// - Store-level write errors.
    async fn close_attempt(
        &self,
        id: &TurnAttemptId,
        params: CloseAttemptParams,
        now: OffsetDateTime,
    ) -> Result<TurnAttempt>;

    /// Look up a single attempt by id.
    ///
    /// Returns `None` if the id does not exist.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn get(&self, id: &TurnAttemptId) -> Result<Option<TurnAttempt>>;

    /// List all attempts for a task, ordered by `attempt_number`
    /// ascending.
    ///
    /// Returns an empty vec if no attempts exist for the task.
    ///
    /// # Errors
    /// Returns an error if the underlying store cannot be queried.
    async fn list_by_task(&self, task_id: &AgentTaskId) -> Result<Vec<TurnAttempt>>;
}

// ─────────────────────────────────────────────────────────────────────
// In-memory reference implementation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Inner {
    /// Primary key index.
    by_id: HashMap<TurnAttemptId, TurnAttempt>,
    /// Secondary index: task → attempt ids, in insertion order
    /// (sorted at query time by `attempt_number`).
    by_task: HashMap<AgentTaskId, Vec<TurnAttemptId>>,
}

/// In-memory reference implementation of [`TurnAttemptStore`].
///
/// Follows the same `Arc<RwLock<Inner>>` pattern as
/// [`super::thread_store::InMemoryThreadStore`]. Designed for tests
/// and single-process usage.
#[derive(Clone, Default)]
pub struct InMemoryTurnAttemptStore {
    inner: Arc<RwLock<Inner>>,
}

impl InMemoryTurnAttemptStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl TurnAttemptStore for InMemoryTurnAttemptStore {
    async fn open_attempt(&self, params: OpenAttemptParams) -> Result<TurnAttempt> {
        let attempt = TurnAttempt::open(params);
        let mut inner = self.inner.write().await;
        inner
            .by_task
            .entry(attempt.task_id.clone())
            .or_default()
            .push(attempt.id.clone());
        inner.by_id.insert(attempt.id.clone(), attempt.clone());
        drop(inner);
        Ok(attempt)
    }

    async fn close_attempt(
        &self,
        id: &TurnAttemptId,
        params: CloseAttemptParams,
        now: OffsetDateTime,
    ) -> Result<TurnAttempt> {
        let mut inner = self.inner.write().await;
        let attempt = inner
            .by_id
            .get(id)
            .ok_or_else(|| anyhow!("attempt not found: {id}"))?
            .clone();

        let closed = attempt.close(params, now)?;
        inner.by_id.insert(closed.id.clone(), closed.clone());
        drop(inner);
        Ok(closed)
    }

    async fn get(&self, id: &TurnAttemptId) -> Result<Option<TurnAttempt>> {
        let inner = self.inner.read().await;
        Ok(inner.by_id.get(id).cloned())
    }

    async fn list_by_task(&self, task_id: &AgentTaskId) -> Result<Vec<TurnAttempt>> {
        let inner = self.inner.read().await;
        let Some(ids) = inner.by_task.get(task_id) else {
            return Ok(Vec::new());
        };
        let mut attempts: Vec<TurnAttempt> = ids
            .iter()
            .filter_map(|id| inner.by_id.get(id).cloned())
            .collect();
        drop(inner);
        attempts.sort_by_key(|a| a.attempt_number);
        Ok(attempts)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::turn_attempt::TurnAttemptSchemaError;
    use super::*;
    use agent_sdk_core::audit::AuditProvenance;
    use anyhow::{Context, Result};
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn open_params(task_id: &str, attempt_number: u32) -> OpenAttemptParams {
        OpenAttemptParams {
            task_id: AgentTaskId::from_string(task_id),
            attempt_number,
            provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
            request_blob: serde_json::json!({"messages": []}),
            now: t0(),
        }
    }

    fn close_params() -> CloseAttemptParams {
        use super::super::turn_attempt::TurnAttemptOutcome;
        CloseAttemptParams {
            response_blob: serde_json::json!({"id": "msg_1"}),
            response_id: Some("msg_1".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(agent_sdk_core::llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 10,
        }
    }

    // ── Open ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn open_attempt_persists_and_is_readable() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        let attempt = store
            .open_attempt(open_params("task_1", 1))
            .await
            .context("open")?;

        assert!(attempt.is_open());
        let fetched = store
            .get(&attempt.id)
            .await
            .context("get")?
            .context("exists")?;
        assert_eq!(fetched.id, attempt.id);
        Ok(())
    }

    #[tokio::test]
    async fn open_multiple_attempts_for_same_task() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        let a1 = store
            .open_attempt(open_params("task_1", 1))
            .await
            .context("open 1")?;
        let a2 = store
            .open_attempt(open_params("task_1", 2))
            .await
            .context("open 2")?;

        assert_ne!(a1.id, a2.id);
        assert_eq!(a1.attempt_number, 1);
        assert_eq!(a2.attempt_number, 2);
        Ok(())
    }

    // ── Close ────────────────────────────────────────────────────

    #[tokio::test]
    async fn close_attempt_fills_response_fields() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        let attempt = store
            .open_attempt(open_params("task_1", 1))
            .await
            .context("open")?;
        let closed = store
            .close_attempt(&attempt.id, close_params(), t_plus(5))
            .await
            .context("close")?;

        assert!(closed.is_closed());
        assert_eq!(closed.duration_ms, Some(5_000));
        assert_eq!(closed.response_id, Some("msg_1".into()));

        // Verify the stored row is also closed.
        let fetched = store
            .get(&attempt.id)
            .await
            .context("get")?
            .context("exists")?;
        assert!(fetched.is_closed());
        Ok(())
    }

    #[tokio::test]
    async fn close_already_closed_returns_error() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        let attempt = store
            .open_attempt(open_params("task_1", 1))
            .await
            .context("open")?;
        store
            .close_attempt(&attempt.id, close_params(), t_plus(1))
            .await
            .context("close")?;

        let err = store
            .close_attempt(&attempt.id, close_params(), t_plus(2))
            .await
            .expect_err("should reject double close");
        let schema_err = err.downcast_ref::<TurnAttemptSchemaError>();
        assert_eq!(schema_err, Some(&TurnAttemptSchemaError::AlreadyClosed));
        Ok(())
    }

    #[tokio::test]
    async fn close_nonexistent_returns_error() {
        let store = InMemoryTurnAttemptStore::new();
        let err = store
            .close_attempt(
                &TurnAttemptId::from_string("attempt_missing"),
                close_params(),
                t_plus(1),
            )
            .await
            .expect_err("should fail");
        assert!(
            err.to_string().contains("not found"),
            "expected 'not found', got: {err}",
        );
    }

    // ── List ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn list_by_task_returns_ordered_attempts() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        // Insert out of order to verify sorting.
        store
            .open_attempt(open_params("task_1", 3))
            .await
            .context("open 3")?;
        store
            .open_attempt(open_params("task_1", 1))
            .await
            .context("open 1")?;
        store
            .open_attempt(open_params("task_1", 2))
            .await
            .context("open 2")?;

        let list = store
            .list_by_task(&AgentTaskId::from_string("task_1"))
            .await
            .context("list")?;
        assert_eq!(list.len(), 3);
        let numbers: Vec<u32> = list.iter().map(|a| a.attempt_number).collect();
        assert_eq!(numbers, vec![1, 2, 3]);
        Ok(())
    }

    #[tokio::test]
    async fn list_by_task_returns_empty_for_unknown_task() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        let list = store
            .list_by_task(&AgentTaskId::from_string("task_unknown"))
            .await
            .context("list")?;
        assert!(list.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn list_by_task_does_not_mix_tasks() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        store
            .open_attempt(open_params("task_a", 1))
            .await
            .context("open a1")?;
        store
            .open_attempt(open_params("task_b", 1))
            .await
            .context("open b1")?;
        store
            .open_attempt(open_params("task_a", 2))
            .await
            .context("open a2")?;

        let list_a = store
            .list_by_task(&AgentTaskId::from_string("task_a"))
            .await
            .context("list a")?;
        assert_eq!(list_a.len(), 2);
        assert!(list_a.iter().all(|a| a.task_id.as_str() == "task_a"));

        let list_b = store
            .list_by_task(&AgentTaskId::from_string("task_b"))
            .await
            .context("list b")?;
        assert_eq!(list_b.len(), 1);
        Ok(())
    }

    // ── Get ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn get_nonexistent_returns_none() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        let result = store
            .get(&TurnAttemptId::from_string("attempt_nope"))
            .await
            .context("get")?;
        assert!(result.is_none());
        Ok(())
    }

    // ── Mixed open/closed ────────────────────────────────────────

    #[tokio::test]
    async fn list_includes_both_open_and_closed_attempts() -> Result<()> {
        let store = InMemoryTurnAttemptStore::new();
        let a1 = store
            .open_attempt(open_params("task_1", 1))
            .await
            .context("open 1")?;
        store
            .close_attempt(&a1.id, close_params(), t_plus(2))
            .await
            .context("close 1")?;
        store
            .open_attempt(open_params("task_1", 2))
            .await
            .context("open 2")?;

        let list = store
            .list_by_task(&AgentTaskId::from_string("task_1"))
            .await
            .context("list")?;
        assert_eq!(list.len(), 2);
        assert!(list[0].is_closed());
        assert!(list[1].is_open());
        Ok(())
    }
}
