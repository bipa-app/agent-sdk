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

    use agent_sdk_foundation::audit::AuditProvenance;
    use agent_sdk_foundation::{ThreadId, TokenUsage};
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
                    stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
                    outcome: TurnAttemptOutcome::Success,
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                },
                messages: vec![agent_sdk_foundation::llm::Message::assistant(
                    "test response",
                )],
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
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
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
                    state: agent_sdk_foundation::AgentState::new(thread_id("conformance-children")),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        };
        let (parent, children) = task_store
            .spawn_tool_children(
                &root.id,
                &worker,
                &lease,
                vec![spec],
                payload,
                None,
                t_plus(62),
            )
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
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
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
                    state: agent_sdk_foundation::AgentState::new(thread_id("conformance-orphan")),
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
                None,
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
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
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
                    state: agent_sdk_foundation::AgentState::new(thread_id("conformance-clear")),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        };
        let (_parent, children) = task_store
            .spawn_tool_children(
                &root.id,
                &worker,
                &lease,
                vec![spec],
                payload,
                None,
                t_plus(102),
            )
            .await?;
        assert_eq!(children.len(), 1);

        // Clear must succeed despite parent→child and child→parent FK edges.
        task_store.clear().await?;
        assert!(task_store.get(&root.id).await?.is_none());
        assert!(task_store.get(&children[0].id).await?.is_none());
        Ok(())
    }

    // ── Phase 10 · E: idempotency + back-pressure ───────────────────

    fn submit_params(
        task: AgentTask,
        request_id: &str,
        fingerprint: &[u8],
        max_queued_depth: Option<u32>,
    ) -> agent_server::journal::store::SubmitRootTurnParams {
        let result_json = serde_json::json!({ "task_id": task.id.to_string() });
        agent_server::journal::store::SubmitRootTurnParams {
            task,
            idempotency: Some(agent_server::journal::store::SubmitRootIdempotency {
                request_id: request_id.to_owned(),
                fingerprint: fingerprint.to_vec(),
                result_json,
            }),
            max_queued_depth,
        }
    }

    /// A retried submission under the same `request_id` admits exactly
    /// one root turn and replays the original on the retry. This is the
    /// at-least-once dedup contract (gap #6).
    async fn test_submit_idempotent_dedup(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = thread_id("conformance-idem");
        let first = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        let first_id = first.id.clone();
        let outcome = task_store
            .submit_root_turn_idempotent(submit_params(first, "req-1", b"fp", None))
            .await
            .map_err(|error| anyhow::anyhow!("first submit failed: {error}"))?;
        assert!(!outcome.replayed, "first submit must not be a replay");
        assert_eq!(outcome.task.id, first_id);

        // A retry with a *different* task row but the same request_id +
        // fingerprint replays the original task and admits nothing new.
        let retry = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let retry_id = retry.id.clone();
        let replay = task_store
            .submit_root_turn_idempotent(submit_params(retry, "req-1", b"fp", None))
            .await
            .map_err(|error| anyhow::anyhow!("retry submit failed: {error}"))?;
        assert!(replay.replayed, "retry must be a replay");
        assert_eq!(replay.task.id, first_id, "replay returns the original task");
        assert_ne!(
            replay.task.id, retry_id,
            "the retry row must not be admitted"
        );

        // Exactly one root task exists on the thread.
        let tasks = task_store.list_by_thread(&tid).await?;
        let roots = tasks.iter().filter(|task| task.is_root()).count();
        assert_eq!(roots, 1, "exactly one root admitted across the retry");
        Ok(())
    }

    /// A retry that re-uses a `request_id` with a *different* payload
    /// fingerprint is a conflict, not a silent alias.
    async fn test_submit_idempotent_conflict(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = thread_id("conformance-idem-conflict");
        let first = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        task_store
            .submit_root_turn_idempotent(submit_params(first, "req-c", b"fp-a", None))
            .await
            .map_err(|error| anyhow::anyhow!("first submit failed: {error}"))?;

        let second = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let result = task_store
            .submit_root_turn_idempotent(submit_params(second, "req-c", b"fp-b", None))
            .await;
        assert!(
            matches!(
                result,
                Err(agent_server::journal::store::SubmitRootTurnError::IdempotencyConflict)
            ),
            "a different fingerprint under the same key must conflict",
        );
        Ok(())
    }

    /// Exceeding the per-thread queued-root depth cap returns
    /// `QueueDepthExceeded` (gap #9). The active root never counts
    /// against the cap; only queued roots do.
    async fn test_submit_queue_depth_cap(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = thread_id("conformance-queue-cap");
        // First admission takes the active slot (queued_depth == 0).
        let active = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        let outcome = task_store
            .submit_root_turn_idempotent(submit_params(active, "q-active", b"fp", Some(1)))
            .await
            .map_err(|error| anyhow::anyhow!("active submit failed: {error}"))?;
        assert_eq!(outcome.queued_depth, 0);

        // Second admission queues behind the active slot (queued_depth
        // becomes 1, which equals the cap — still allowed because the
        // check is "would exceed", i.e. current >= cap before insert).
        let queued = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let outcome = task_store
            .submit_root_turn_idempotent(submit_params(queued, "q-1", b"fp", Some(1)))
            .await
            .map_err(|error| anyhow::anyhow!("queued submit failed: {error}"))?;
        assert_eq!(
            outcome.queued_depth, 1,
            "one queued root after second admit"
        );

        // Third admission would push queued depth to 2 > cap of 1.
        let overflow = AgentTask::new_root_turn(tid.clone(), t_plus(12), 3);
        let result = task_store
            .submit_root_turn_idempotent(submit_params(overflow, "q-2", b"fp", Some(1)))
            .await;
        match result {
            Err(agent_server::journal::store::SubmitRootTurnError::QueueDepthExceeded {
                cap,
                current_depth,
            }) => {
                assert_eq!(cap, 1);
                assert_eq!(current_depth, 1);
            }
            other => anyhow::bail!("expected QueueDepthExceeded, got {other:?}"),
        }

        // The rejected submission left no trace — still exactly two roots.
        let tasks = task_store.list_by_thread(&tid).await?;
        let roots = tasks.iter().filter(|task| task.is_root()).count();
        assert_eq!(roots, 2, "the over-cap submission admitted nothing");
        Ok(())
    }

    /// A root admitted with the production budget that crashes mid-turn
    /// is requeued (not failed-closed). This is the `max_attempts`
    /// reconciliation (gap #10): `DEFAULT_ROOT_MAX_ATTEMPTS == 3`.
    async fn test_root_default_budget_requeues_on_crash(
        task_store: &dyn AgentTaskStore,
    ) -> Result<()> {
        assert_eq!(
            AgentTask::DEFAULT_ROOT_MAX_ATTEMPTS,
            3,
            "root default budget must allow a mid-turn crash to retry",
        );
        let tid = thread_id("conformance-root-budget");
        let root = AgentTask::new_root_turn(
            tid.clone(),
            t_plus(10),
            AgentTask::DEFAULT_ROOT_MAX_ATTEMPTS,
        );
        let _ = task_store.submit_root_turn(root.clone()).await?;

        // Acquire (attempt 1) and let the lease expire — simulating a
        // mid-turn host crash.
        let _ = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(11),
                t_plus(10),
            )
            .await?;
        let records = task_store.release_expired_leases(t_plus(12)).await?;
        assert_eq!(records.len(), 1);
        assert!(
            records[0].action.is_requeue(),
            "budget-3 root must requeue on first crash, got {:?}",
            records[0].action,
        );

        let task = task_store.get(&root.id).await?.context("task exists")?;
        assert_eq!(task.status, TaskStatus::Pending, "requeued back to Pending");
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

    #[tokio::test]
    async fn conformance_in_memory_submit_idempotent_dedup() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_submit_idempotent_dedup(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_submit_idempotent_conflict() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_submit_idempotent_conflict(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_submit_queue_depth_cap() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_submit_queue_depth_cap(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_root_default_budget_requeues() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_root_default_budget_requeues_on_crash(s.task.as_ref()).await
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

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_submit_idempotent_dedup() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_submit_idempotent_dedup(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_submit_idempotent_conflict() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_submit_idempotent_conflict(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_submit_queue_depth_cap() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_submit_queue_depth_cap(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_root_default_budget_requeues() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_root_default_budget_requeues_on_crash(&store).await
    }

    /// Cross-restart idempotency (gap #6, the headline acceptance
    /// criterion): submit the same `request_id` before and after a
    /// simulated host restart — modeled by dropping the `SQLite` store +
    /// pool and reopening the *same file* — and assert exactly one
    /// `RootTurn` task exists, with the retry replaying the original.
    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn cross_restart_submit_idempotency_admits_exactly_one_root() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let db_path = dir.path().join("restart.db");
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let tid = thread_id("restart-idem");

        // ── Before the "restart" ─────────────────────────────────────
        let original_task_id = {
            let store = crate::sqlite::SqliteDurableStore::connect(&url).await?;
            let root = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
            let task_id = root.id.clone();
            let outcome = store
                .submit_root_turn_idempotent(submit_params(root, "restart-req", b"fp", None))
                .await
                .map_err(|error| anyhow::anyhow!("pre-restart submit failed: {error}"))?;
            assert!(!outcome.replayed);
            // Drop the store + pool to simulate process death.
            drop(store);
            task_id
        };

        // ── After the "restart": reopen the same file ────────────────
        let store = crate::sqlite::SqliteDurableStore::connect(&url).await?;
        // The in-memory dedup cache is gone, but the durable record
        // survives. A retry under the same request_id must replay the
        // original task, not admit a duplicate.
        let retry = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let replay = store
            .submit_root_turn_idempotent(submit_params(retry, "restart-req", b"fp", None))
            .await
            .map_err(|error| anyhow::anyhow!("post-restart retry failed: {error}"))?;
        assert!(replay.replayed, "post-restart retry must replay, not admit");
        assert_eq!(
            replay.task.id, original_task_id,
            "replay returns the original task minted before the restart",
        );

        let tasks = AgentTaskStore::list_by_thread(&store, &tid).await?;
        let roots = tasks.iter().filter(|task| task.is_root()).count();
        assert_eq!(
            roots, 1,
            "exactly one RootTurn task survives the request_id retry across restart",
        );
        Ok(())
    }
}
