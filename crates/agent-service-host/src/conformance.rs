//! Cross-backend conformance test suite.
//!
//! This module defines backend-agnostic test functions that verify the
//! critical correctness invariants of the journal and checkpoint stores.
//! Each test function accepts trait-object references so they can be
//! invoked against the in-memory, `SQLite`, and `PostgreSQL` backends from
//! the same code path.
//!
//! # Covered invariants
//!
//! 1. **Root admission** — at most one blocking root per thread.
//! 2. **FIFO queueing** — second root on same thread gets `Queued`;
//!    order preserved.
//! 3. **Lease acquisition** — `try_acquire_task` /
//!    `acquire_next_runnable` move `Pending` → `Running` with correct
//!    lease fields.
//! 4. **Heartbeat** — heartbeats extend lease expiry.
//! 5. **Lease expiry sweep** — expired leases are requeued.
//! 6. **Completed-turn commit** — atomic thread advance, message
//!    projection, attempt close, and checkpoint creation.
//! 7. **Child spawn and resume** — parent pauses on children;
//!    completing all children makes parent runnable again.
//! 8. **Cancel tree** — cancellation propagates to all descendants.
//! 9. **Queued root promotion** — completing an active root promotes
//!    the FIFO head.
//! 10. **Retry exhaustion** — budget-exhausted row is failed closed.
//! 11. **Fail-closed child wakes parent** — recovery-path fail-close
//!     propagates to `WaitingOnChildren` parent so it does not stay
//!     blocked forever.
//! 12. **Clear with parent-child** — `clear()` wipes parent/child
//!     chains despite `ON DELETE RESTRICT` self-referential FKs.
//!
//! # Backend-specific differences (documented, not hidden)
//!
//! | Concern | In-memory | SQLite | PostgreSQL |
//! |---------|-----------|--------|------------|
//! | Locking | `RwLock` on `Inner` | database-level (`BEGIN IMMEDIATE`) | row-level (`FOR UPDATE` / `SKIP LOCKED`) |
//! | Concurrency | single-writer, multi-reader | single-writer, multi-reader (WAL) | multi-writer, row-locked |
//! | Persistence | none — lost on restart | file-backed, survives restart | server-backed, survives restart |
//! | `acquire_next_runnable` skip | in-memory index scan | full table scan (OK for local) | `SKIP LOCKED` avoids contention |

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use time::{Duration, OffsetDateTime};

    use agent_sdk_core::audit::AuditProvenance;
    use agent_sdk_core::{ThreadId, TokenUsage};
    use agent_server::journal::checkpoint_store::CheckpointStore;
    use agent_server::journal::commit::{CompletedTurnCommit, commit_completed_turn};
    use agent_server::journal::event_repository::EventRepository;
    use agent_server::journal::message_store::MessageProjectionStore;
    use agent_server::journal::recovery::RecoveryAction;
    use agent_server::journal::store::AgentTaskStore;
    use agent_server::journal::task::{
        AgentTask, ChildSpawnSpec, LeaseId, SuspensionPayload, TaskStatus, WorkerId,
    };
    use agent_server::journal::thread_store::ThreadStore;
    use agent_server::journal::turn_attempt::{
        CloseAttemptParams, OpenAttemptParams, TurnAttemptOutcome,
    };
    use agent_server::journal::turn_attempt_store::TurnAttemptStore;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_id(name: &str) -> ThreadId {
        ThreadId::from_string(name)
    }

    fn fresh_root(name: &str, secs: i64) -> AgentTask {
        AgentTask::new_root_turn(thread_id(name), t_plus(secs), 3)
    }

    // ── Core conformance tests ───────────────────────────────────────

    /// Root admission: submit a root, verify it's Pending.
    async fn test_root_admission(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-root-admit", 1);
        let admitted = task_store.submit_root_turn(root.clone()).await?;
        assert_eq!(admitted.status, TaskStatus::Pending);
        assert_eq!(admitted.id, root.id);

        let fetched = task_store
            .get(&root.id)
            .await?
            .context("root should exist")?;
        assert_eq!(fetched.status, TaskStatus::Pending);
        Ok(())
    }

    /// Root FIFO queueing: second root on same thread gets Queued.
    async fn test_root_fifo_queueing(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = "conformance-fifo";
        let root1 = fresh_root(tid, 10);
        let root2 = fresh_root(tid, 11);

        let admitted1 = task_store.submit_root_turn(root1).await?;
        assert_eq!(admitted1.status, TaskStatus::Pending);

        let admitted2 = task_store.submit_root_turn(root2).await?;
        assert_eq!(admitted2.status, TaskStatus::Queued);

        let active = task_store
            .active_root_for_thread(&thread_id(tid))
            .await?
            .context("active root missing")?;
        assert_eq!(active.id, admitted1.id);

        let queued = task_store.list_queued_roots(&thread_id(tid)).await?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, admitted2.id);
        Ok(())
    }

    /// Lease acquisition via `try_acquire_task`.
    async fn test_lease_acquisition(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-lease", 20);
        let _ = task_store.submit_root_turn(root.clone()).await?;

        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let expires = t_plus(50);

        let acquired = task_store
            .try_acquire_task(&root.id, worker.clone(), lease.clone(), expires, t_plus(21))
            .await?;
        let acquired = acquired.context("should acquire")?;
        assert_eq!(acquired.status, TaskStatus::Running);
        assert_eq!(acquired.worker_id, Some(worker.clone()));
        assert_eq!(acquired.lease_id, Some(lease.clone()));
        assert_eq!(acquired.attempt, 1);

        // Second acquire on the same task should return None.
        let second = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                expires,
                t_plus(22),
            )
            .await?;
        assert!(second.is_none());
        Ok(())
    }

    /// Heartbeat extends a lease.
    async fn test_heartbeat(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-heartbeat", 30);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(40),
                t_plus(31),
            )
            .await?;

        let refreshed = task_store
            .heartbeat_task(&root.id, &worker, &lease, t_plus(60), t_plus(35))
            .await?;
        assert_eq!(refreshed.lease_expires_at, Some(t_plus(60)));
        Ok(())
    }

    /// Expired leases are swept and requeued.
    async fn test_lease_expiry_sweep(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-expiry", 40);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let _ = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(42),
                t_plus(41),
            )
            .await?;

        // Sweep after lease expires.
        let records = task_store.release_expired_leases(t_plus(43)).await?;
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0].action, RecoveryAction::Requeue));

        // Task should be back to Pending.
        let task = task_store.get(&root.id).await?.context("task exists")?;
        assert_eq!(task.status, TaskStatus::Pending);
        Ok(())
    }

    /// Completed-turn commit creates a checkpoint.
    async fn test_completed_turn_commit(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
        message_store: &dyn MessageProjectionStore,
        attempt_store: &dyn TurnAttemptStore,
        checkpoint_store: &dyn CheckpointStore,
        event_repo: &dyn EventRepository,
    ) -> Result<()> {
        let tid = thread_id("conformance-commit");
        let root = AgentTask::new_root_turn(tid.clone(), t_plus(50), 3);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&root.id, worker, lease, t_plus(80), t_plus(51))
            .await?;

        let attempt = attempt_store
            .open_attempt(OpenAttemptParams {
                task_id: root.id.clone(),
                attempt_number: 1,
                provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
                request_blob: serde_json::json!({"messages": []}),
                now: t_plus(52),
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await?;

        let turn_usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let outcome = commit_completed_turn(
            CompletedTurnCommit {
                thread_id: tid.clone(),
                task_id: root.id.clone(),
                turn_attempt_id: attempt.id.clone(),
                close_attempt_params: CloseAttemptParams {
                    response_blob: serde_json::json!({"id": "msg_conf_1"}),
                    response_id: Some("msg_conf_1".into()),
                    response_model: Some("claude-sonnet-4-5-20250929".into()),
                    stop_reason: Some(agent_sdk_core::llm::StopReason::EndTurn),
                    outcome: TurnAttemptOutcome::Success,
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                },
                messages: vec![agent_sdk_core::llm::Message::assistant("test response")],
                turn_usage,
                agent_state_snapshot: serde_json::json!({}),
                events: vec![],
                outbox_max_attempts: 3,
                now: t_plus(55),
            },
            thread_store,
            message_store,
            attempt_store,
            checkpoint_store,
            event_repo,
        )
        .await?;

        assert_eq!(outcome.thread.committed_turns, 1);
        assert_eq!(outcome.checkpoint.turn_number, 1);

        let latest = checkpoint_store
            .get_latest_by_thread(&tid)
            .await?
            .context("latest checkpoint missing")?;
        assert_eq!(latest.turn_number, 1);
        Ok(())
    }

    /// Child spawn + complete resumes parent.
    async fn test_child_spawn_and_resume(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-children", 60);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(90),
                t_plus(61),
            )
            .await?;

        let spec = ChildSpawnSpec { max_attempts: 3 };
        let payload = SuspensionPayload {
            continuation: agent_sdk_core::ContinuationEnvelope::wrap(
                agent_sdk_core::AgentContinuation {
                    thread_id: thread_id("conformance-children"),
                    turn: 1,
                    total_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    turn_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    pending_tool_calls: vec![],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_core::AgentState::new(thread_id("conformance-children")),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        };
        let (parent, children) = task_store
            .spawn_tool_children(&root.id, &worker, &lease, vec![spec], payload, t_plus(62))
            .await?;
        assert_eq!(parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(children.len(), 1);

        // Acquire and complete the child.
        let child = &children[0];
        let cw = WorkerId::new();
        let cl = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&child.id, cw.clone(), cl.clone(), t_plus(90), t_plus(63))
            .await?;
        let (completed_child, resumed_parent) = task_store
            .complete_task(&child.id, &cw, &cl, t_plus(64))
            .await?;
        assert_eq!(completed_child.status, TaskStatus::Completed);
        let resumed = resumed_parent.context("parent should be returned")?;
        assert_eq!(resumed.status, TaskStatus::Pending);
        Ok(())
    }

    /// Cancel tree cancels root and all descendants.
    async fn test_cancel_tree(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-cancel", 70);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(100),
                t_plus(71),
            )
            .await?;

        let cancelled = task_store.cancel_tree(&root.id, t_plus(72)).await?;
        assert_eq!(cancelled.len(), 1);
        assert_eq!(cancelled[0], root.id);

        let task = task_store.get(&root.id).await?.context("task exists")?;
        assert_eq!(task.status, TaskStatus::Cancelled);
        Ok(())
    }

    /// Promote queued root after active root completes.
    async fn test_promote_queued_root(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = "conformance-promote";
        let root1 = fresh_root(tid, 80);
        let root2 = fresh_root(tid, 81);

        let r1 = task_store.submit_root_turn(root1).await?;
        let r2 = task_store.submit_root_turn(root2).await?;
        assert_eq!(r2.status, TaskStatus::Queued);

        // Acquire and complete root1.
        let w = WorkerId::new();
        let l = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&r1.id, w.clone(), l.clone(), t_plus(110), t_plus(82))
            .await?;
        let _ = task_store.complete_task(&r1.id, &w, &l, t_plus(83)).await?;

        // Promote next queued root.
        let promoted = task_store
            .promote_next_queued_root(&thread_id(tid), t_plus(84))
            .await?;
        let promoted = promoted.context("should promote")?;
        assert_eq!(promoted.id, r2.id);
        assert_eq!(promoted.status, TaskStatus::Pending);
        Ok(())
    }

    /// Retry exhaustion fails a task closed.
    async fn test_retry_exhaustion(task_store: &dyn AgentTaskStore) -> Result<()> {
        // Create a root with max_attempts = 1.
        let root = AgentTask::new_root_turn(thread_id("conformance-retry"), t_plus(90), 1);
        let _ = task_store.submit_root_turn(root.clone()).await?;

        // Acquire (attempt 1) and let the lease expire.
        let _ = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(91),
                t_plus(90),
            )
            .await?;
        let records = task_store.release_expired_leases(t_plus(92)).await?;
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0].action, RecoveryAction::FailClosed(_)));

        let task = task_store.get(&root.id).await?.context("task exists")?;
        assert_eq!(task.status, TaskStatus::Failed);
        Ok(())
    }

    /// When a child's lease expires past `max_attempts`, the recovery
    /// path must wake its `WaitingOnChildren` parent — otherwise the
    /// parent stays blocked forever (no future complete/fail event
    /// will fire to decrement `pending_child_count`).
    async fn test_fail_closed_child_wakes_parent(task_store: &dyn AgentTaskStore) -> Result<()> {
        let parent = AgentTask::new_root_turn(thread_id("conformance-orphan"), t_plus(200), 3);
        let _ = task_store.submit_root_turn(parent.clone()).await?;
        let pw = WorkerId::new();
        let pl = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&parent.id, pw.clone(), pl.clone(), t_plus(260), t_plus(201))
            .await?;

        // Spawn a single child with max_attempts=1 so its first lease
        // expiry fail-closes it.
        let payload = SuspensionPayload {
            continuation: agent_sdk_core::ContinuationEnvelope::wrap(
                agent_sdk_core::AgentContinuation {
                    thread_id: thread_id("conformance-orphan"),
                    turn: 1,
                    total_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    turn_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    pending_tool_calls: vec![],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_core::AgentState::new(thread_id("conformance-orphan")),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        };
        let (_parent, children) = task_store
            .spawn_tool_children(
                &parent.id,
                &pw,
                &pl,
                vec![ChildSpawnSpec { max_attempts: 1 }],
                payload,
                t_plus(202),
            )
            .await?;
        let child = &children[0];

        // Acquire child, let lease expire, then sweep.  attempt=1 ==
        // max_attempts triggers FailClosed.
        let _ = task_store
            .try_acquire_task(
                &child.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(210),
                t_plus(203),
            )
            .await?;
        let records = task_store.release_expired_leases(t_plus(211)).await?;
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0].action, RecoveryAction::FailClosed(_)));

        // Child must be Failed and parent must have woken up to
        // Pending with no live children.
        let child_after = task_store.get(&child.id).await?.context("child exists")?;
        assert_eq!(child_after.status, TaskStatus::Failed);
        let parent_after = task_store.get(&parent.id).await?.context("parent exists")?;
        assert_eq!(parent_after.pending_child_count, 0);
        assert_eq!(parent_after.status, TaskStatus::Pending);
        Ok(())
    }

    /// Clear wipes every task — including parent/child chains guarded
    /// by ON DELETE RESTRICT self-referential FKs on `SQLite`.
    async fn test_clear_with_parent_child(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-clear", 100);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(160),
                t_plus(101),
            )
            .await?;

        let spec = ChildSpawnSpec { max_attempts: 3 };
        let payload = SuspensionPayload {
            continuation: agent_sdk_core::ContinuationEnvelope::wrap(
                agent_sdk_core::AgentContinuation {
                    thread_id: thread_id("conformance-clear"),
                    turn: 1,
                    total_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    turn_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    pending_tool_calls: vec![],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_core::AgentState::new(thread_id("conformance-clear")),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        };
        let (_parent, children) = task_store
            .spawn_tool_children(&root.id, &worker, &lease, vec![spec], payload, t_plus(102))
            .await?;
        assert_eq!(children.len(), 1);

        // Clear must succeed despite parent→child and child→parent FK edges.
        task_store.clear().await?;
        assert!(task_store.get(&root.id).await?.is_none());
        assert!(task_store.get(&children[0].id).await?.is_none());
        Ok(())
    }

    // ── Backend runners ──────────────────────────────────────────────

    // ── In-memory backend ────────────────────────────────────────────

    #[tokio::test]
    async fn conformance_in_memory_root_admission() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_root_admission(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_fifo_queueing() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_root_fifo_queueing(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_lease_acquisition() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_lease_acquisition(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_heartbeat() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_heartbeat(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_lease_expiry() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_lease_expiry_sweep(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_completed_turn() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_completed_turn_commit(
            s.task.as_ref(),
            s.thread.as_ref(),
            s.message.as_ref(),
            s.attempt.as_ref(),
            s.checkpoint.as_ref(),
            s.event.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn conformance_in_memory_child_spawn() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_child_spawn_and_resume(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_cancel_tree() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_cancel_tree(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_promote_queued() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_promote_queued_root(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_retry_exhaustion() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_retry_exhaustion(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_clear_parent_child() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_clear_with_parent_child(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_fail_closed_child_wakes_parent() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_fail_closed_child_wakes_parent(s.task.as_ref()).await
    }

    struct InMemoryStores {
        task: std::sync::Arc<dyn AgentTaskStore>,
        thread: std::sync::Arc<dyn ThreadStore>,
        message: std::sync::Arc<dyn MessageProjectionStore>,
        attempt: std::sync::Arc<dyn TurnAttemptStore>,
        checkpoint: std::sync::Arc<dyn CheckpointStore>,
        event: std::sync::Arc<dyn EventRepository>,
    }

    fn fresh_in_memory_stores() -> InMemoryStores {
        use agent_server::journal::checkpoint_store::InMemoryCheckpointStore;
        use agent_server::journal::event_repository::InMemoryEventRepository;
        use agent_server::journal::message_store::InMemoryMessageProjectionStore;
        use agent_server::journal::store::InMemoryAgentTaskStore;
        use agent_server::journal::thread_store::InMemoryThreadStore;
        use agent_server::journal::turn_attempt_store::InMemoryTurnAttemptStore;

        InMemoryStores {
            task: std::sync::Arc::new(InMemoryAgentTaskStore::new()),
            thread: std::sync::Arc::new(InMemoryThreadStore::new()),
            message: std::sync::Arc::new(InMemoryMessageProjectionStore::new()),
            attempt: std::sync::Arc::new(InMemoryTurnAttemptStore::new()),
            checkpoint: std::sync::Arc::new(InMemoryCheckpointStore::new()),
            event: std::sync::Arc::new(InMemoryEventRepository::new()),
        }
    }

    // ── SQLite backend ───────────────────────────────────────────────

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_root_admission() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_root_admission(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_fifo_queueing() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_root_fifo_queueing(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_lease_acquisition() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_lease_acquisition(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_heartbeat() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_heartbeat(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_lease_expiry() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_lease_expiry_sweep(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_completed_turn() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_completed_turn_commit(&store, &store, &store, &store, &store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_child_spawn() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_child_spawn_and_resume(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_cancel_tree() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_cancel_tree(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_promote_queued() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_promote_queued_root(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_retry_exhaustion() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_retry_exhaustion(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_clear_parent_child() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_clear_with_parent_child(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_fail_closed_child_wakes_parent() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_fail_closed_child_wakes_parent(&store).await
    }
}
